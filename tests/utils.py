#!/usr/bin/env python
"""
A few utility function and classes to simplify playing back and reading audio
streams.
"""
from __future__ import print_function
try:
    import Queue as queue
except ImportError:
    import queue
try:
    basestring
except NameError:
    basestring = (str, bytes)
import threading
import time
import sys
import sounddevice as sd
import soundfile as sf
import traceback


TXQSIZE = 1<<16 # Number of frames to buffer for transmission


# TODO
# Replace Queue with a portaudio ring buffer in order to allow variable block
# sizes
class QueuedStreamBase(sd._StreamBase):
    """
    This class adds a python Queue for reading and writing audio data. This
    double buffers the audio data so that any processing is kept out of the time
    sensitive audio callback function. For maximum flexibility, receive queue
    data is a bytearray object; transmit queue data should be of a buffer type
    where each element is a single byte.

    Notes:
    
    Using a queue requires that we use a fixed, predetermined value for the
    portaudio blocksize.

    If the receive buffer fills or the transmit buffer is found empty during a
    callback the audio stream is aborted and an exception is raised.

    During recording, the end of the stream is signaled by a 'None' value placed
    into the receive queue.

    During playback, the end of the stream is signaled by an item on the queue
    that is smaller than blocksize*channels*samplesize bytes.

    """
    def __init__(self, blocksize=None, qsize=-1, kind='duplex', **kwargs):
        if kwargs.get('callback', None) is None: 
            if kind == 'input':
                kwargs['callback'] = self.icallback
            elif kind == 'output':
                kwargs['callback'] = self.ocallback
            else:
                kwargs['callback'] = self.iocallback

        if (kind == 'output' or kind == 'duplex') and not blocksize:
            raise ValueError("Non-zero blocksize is required for playback mode.")

        kwargs['wrap_callback'] = 'buffer'
        super(QueuedStreamBase, self).__init__(kind=kind, blocksize=blocksize, **kwargs)

        self.status = sd.CallbackFlags()
        self.frame_count = 0
        self._closed = False

        if isinstance(self._device, int):
            self._devname = sd.query_devices(self._device)['name']
        else:
            self._devname = tuple(sd.query_devices(dev)['name'] for dev in self._device)

        if kind == 'duplex':
            self.framesize = self.samplesize[0]*self.channels[0], self.samplesize[1]*self.channels[1]
        else:
            self.framesize = self.samplesize*self.channels

        if kind == 'duplex' or kind == 'output':
            if self.blocksize and qsize > 0:
                qsize = (qsize+self.blocksize-1)//self.blocksize
            self.txq = queue.Queue(qsize)
        else:
            self.txq = None

        if kind == 'duplex' or kind == 'input':
            self.rxq = queue.Queue(-1)
        else:
            self.rxq = None

    def icallback(self, in_data, frame_count, time_info, status):
        self.status |= status
        if status._flags&0xF:
            self._set_exception(sd.PortAudioError(str(status)))
            raise sd.CallbackAbort

        try:
            self.rxq.put_nowait(bytearray(in_data))
        except queue.Full:
            self._set_exception(queue.Full("Receive queue is full."))
            raise sd.CallbackAbort

        self.frame_count += frame_count

    def ocallback(self, out_data, frame_count, time_info, status):
        self.status |= status
        if status._flags&0xF:
            self._set_exception(sd.PortAudioError(str(status)))
            raise sd.CallbackAbort

        try:
            txbuff = self.txq.get_nowait()
        except queue.Empty:
            self._set_exception(queue.Empty("Transmit queue is empty."))
            raise sd.CallbackAbort

        out_data[:len(txbuff)] = txbuff

        # This is our last callback!
        if len(txbuff) < frame_count*self.framesize:
            self.frame_count += len(txbuff)//self.framesize
            raise sd.CallbackStop

        self.frame_count += frame_count

    def iocallback(self, in_data, out_data, frame_count, time_info, status):
        self.status |= status
        if status._flags&0xF:
            self._set_exception(sd.PortAudioError(str(status)))
            raise sd.CallbackAbort

        try:
            txbuff = self.txq.get_nowait()
        except queue.Empty:
            self._set_exception(queue.Empty("Transmit queue is empty."))
            raise sd.CallbackAbort

        out_data[:len(txbuff)] = txbuff

        try:
            self.rxq.put_nowait(bytearray(in_data))
        except queue.Full:
            self._set_exception(queue.Full("Receive queue is full."))
            raise sd.CallbackAbort

        # This is our last callback!
        if len(txbuff) < frame_count*self.framesize[1]:
            self.frame_count += len(txbuff)//self.framesize[1]
            raise sd.CallbackStop

        self.frame_count += frame_count

    def _closequeues(self):
        if not self._closed:
            self._closed = True
        else:
            return

        if self.rxq is not None:
            self.rxq.queue.clear()
            self.rxq.put(None)

        if self.txq is not None:
            with self.txq.mutex:
                self._exit.set()
                self.txq.queue.clear()
                self.txq.not_full.notify()

    def abort(self):
        super(QueuedStreamBase, self).abort()
        self._closequeues()

    def stop(self):
        super(QueuedStreamBase, self).stop()
        self._closequeues()

    def close(self):
        super(QueuedStreamBase, self).close()
        self._closequeues()

    def __repr__(self):
        return ("{0}({1._devname!r}, samplerate={1._samplerate:.0f}, "
                "channels={1._channels}, dtype={1._dtype!r}, blocksize={1._blocksize})").format(self.__class__.__name__, self)

class ThreadedStreamBase(QueuedStreamBase):
    """
    This class builds on the QueuedStream class by adding the ability to
    register functions for reading (qreader) and writing (qwriter) audio data
    which run in their own threads. However, the qreader and qwriter threads are
    optional; this allows the use of a 'duplex' stream which e.g. has a
    dedicated thread for writing data but for which data is read in the main
    thread.

    An important advantage with this class is that it properly handles any
    exceptions raised in the qreader and qwriter threads. Specifcally, if an
    exception is raised, the stream will be aborted and the exception re-raised
    in the main thread.
    """
    def __init__(self, blocksize=None, qreader=None, qwriter=None, kind='duplex', **kwargs):
        super(ThreadedStreamBase, self).__init__(kind=kind, blocksize=blocksize, **kwargs)

        self._exit = threading.Event()
        self._exc = queue.Queue()

        if (kind == 'duplex' or kind == 'output') and qwriter is not None:
            txt = threading.Thread(target=self._qrwwrapper, args=(self.txq, qwriter))
            txt.daemon = True
            self.txt = txt
        else:
            self.txt = None

        if (kind == 'duplex' or kind == 'input') and qreader is not None:
            rxt = threading.Thread(target=self._qrwwrapper, args=(self.rxq, qreader))
            rxt.daemon = True
            self.rxt = rxt
        else:
            self.rxt = None

    def _raise_exceptions(self):
        if self._exc.empty():
            return

        exc = self._exc.get()
        if isinstance(exc, tuple):
            exctype, excval, exctb = exc
            # raise exctype, excval, exctb
            try:
                raise exctype(excval).with_traceback(exctb)
            except AttributeError:
                raise exctype(excval)

        raise exc

    def _set_exception(self, exc=None):
        # ignore subsequent exceptions
        if not self._exc.empty():
            return
        if exc is None:
            exc = sys.exc_info()
        self._exc.put(exc)

    def _qrwwrapper(self, queue, qrwfunc):
        """
        Wrapper function for the qreader and qwriter threads which acts as a
        kind of context manager.
        """
        try:                
            qrwfunc(self, queue)
        except:        
            # Raise the exception in the main thread    
            self._set_exception()

            # suppress the exception in this child thread
            try:    self.abort()
            except: pass

    def _stopiothreads(self):
        currthread = threading.current_thread()

        if self.rxt is not None and self.rxt.is_alive() and self.rxt != currthread:
            self.rxt.join()

        if self.txt is not None and self.txt.is_alive() and self.txt != currthread:
            self.txt.join()

    def start(self):
        if self.txt is not None:
            # Prime the buffer
            self.txt.start()
            while not self.txq.full() and self.txt.is_alive():
                time.sleep(0.001)

        if self.rxt is not None:
            self.rxt.start()
        super(ThreadedStreamBase, self).start()

    def abort(self):
        super(ThreadedStreamBase, self).abort()
        self._stopiothreads()

    def stop(self):
        super(ThreadedStreamBase, self).stop()
        self._stopiothreads()

    def close(self):
        super(ThreadedStreamBase, self).close()
        self._stopiothreads()
        self._raise_exceptions()

def _soundfilewriter(stream, rxq):
    if isinstance(stream.dtype, basestring):
        dtype = stream.dtype
    else:
        dtype = stream.dtype[0]

    while True:
        try:
            item = rxq.get(timeout=1)
        except queue.Empty:
            raise queue.Empty("Timed out waiting for data.")
        if item is None: break

        stream.out_fh.buffer_write(item, dtype=dtype)

def _soundfilereader(stream, txq):
    try:               
        framesize = stream.framesize[1]
        dtype = stream.dtype[1]
    except TypeError: 
        framesize = stream.framesize
        dtype = stream.dtype    

    buff = bytearray(stream.fileblocksize*framesize)
    while not stream._exit.is_set():
        nframes = stream.inp_fh.buffer_read_into(buff, dtype=dtype)

        txq.put(bytearray(buff[:nframes*framesize]))

        if nframes < stream.blocksize:
            break

class SoundFileStreamBase(ThreadedStreamBase):
    """
    This helper class basically gives you two things: 

        1) it provides complete qreader and qwriter functions for SoundFile
           objects (or anything that can be opened as a SoundFile object)

        2) it automatically sets parameters for the stream based on the input
           file and automatically sets parameters for the output file based on
           the output stream.
    """
    def __init__(self, inpf=None, outf=None, fileblocksize=None, qreader=None, qwriter=None, kind='duplex', **kwargs):
        # We're playing an audio file, so we can safely assume there's no need
        # to clip
        if kwargs.get('clip_off', None) is None: 
            kwargs['clip_off'] = True

        if kwargs.get('blocksize', None) is None:
            kwargs['blocksize'] = fileblocksize

        self._inpf = inpf
        if inpf is not None and not isinstance(inpf, sf.SoundFile):
            inp_fh = sf.SoundFile(inpf)
        else:
            inp_fh = inpf

        if isinstance(inp_fh, sf.SoundFile):
            if kwargs.get('samplerate', None) is None: 
                kwargs['samplerate'] = inp_fh.samplerate
            if kwargs.get('channels', None) is None: 
                kwargs['channels'] = inp_fh.channels
            if qwriter is None:
                qwriter = _soundfilereader
        # elif inpf is not None and qwriter is None:
        #     raise ValueError("If not using a SoundFile object for playback you must provide your own qwriter routine.")

        self._outf = outf
        try:
            outext = getattr(outf, 'name', outf).rsplit('.', 1)[1].lower()
        except AttributeError:
            outext = None
        if not isinstance(outf, sf.SoundFile) and outf is not None:
            sfkwargs = kwargs['sfkwargs']
            if isinstance(inp_fh, sf.SoundFile):
                if sfkwargs.get('endian', None) is None:
                    sfkwargs['endian'] = inp_fh.endian
                if (sfkwargs.get('format', outext) == inp_fh.format.lower()
                    and sfkwargs.get('subtype', None) is None):
                    sfkwargs['subtype'] = inp_fh.subtype
            if sfkwargs.get('channels', None) is None: 
                sfkwargs['channels'] = self.channels[0]
            if sfkwargs.get('samplerate', None) is None:
                sfkwargs['samplerate'] = int(self.samplerate)
            if sfkwargs.get('mode', None) is None:
                sfkwargs['mode'] = 'w+b'
            out_fh = sf.SoundFile(outf, **sfkwargs)
            if qreader is None:
                qreader = _soundfilewriter                
        else:
            out_fh = outf

        # if outf is not None and qreader is None:
        #     raise ValueError("If not using a SoundFile object for recording you must provide your own qreader routine.")

        super(SoundFileStreamBase, self).__init__(qreader=qreader,
                                                  qwriter=qwriter, 
                                                  kind=kind, **kwargs)

        self.inp_fh = inp_fh
        self.out_fh = outf
        self.fileblocksize = fileblocksize or self.blocksize

    def close(self):
        try:
            super(SoundFileStreamBase, self).close()
        finally:
            if self._outf != self.out_fh:
                self.out_fh.close()
            if self._inpf != self.inp_fh:
                self.inp_fh.close()

class SoundFileInputStream(SoundFileStreamBase):
    def __init__(self, outf, sfkwargs={}, **kwargs):
        super(SoundFileInputStream, self).__init__(outf=outf, kind='input', **kwargs)

class SoundFileOutputStream(SoundFileStreamBase):
    def __init__(self, inpf, qsize=TXQSIZE, fileblocksize=None, **kwargs):
        super(SoundFileOutputStream, self).__init__(inpf=inpf, qsize=qsize,
                                                    kind='output',
                                                    fileblocksize=fileblocksize,
                                                    **kwargs)

class SoundFileStream(SoundFileStreamBase):
    """
    Note that only one of inpf and outf is required. This allows you to e.g. use
    a SoundFile as input but implement your own qreader and/or read from the
    queue in the main thread.
    """
    def __init__(self, inpf=None, outf=None, qsize=TXQSIZE, sfkwargs={}, fileblocksize=None, **kwargs):
        # If you're not using soundfiles for the input or the output, then you
        # should probably be using the Queued or ThreadedStream class
        if inpf is None and outf is None: 
            raise ValueError("No input or output file given.")

        super(SoundFileStream, self).__init__(inpf=inpf, outf=outf, qsize=qsize,
                                              fileblocksize=fileblocksize,
                                              kind='duplex', **kwargs)

def blockstream(inpf=None, blocksize=1024, overlap=0, always_2d=False, copy=False, **kwargs):
    import numpy as np
    assert blocksize is not None and blocksize > 0, "Requires a fixed known blocksize"

    incframes = blocksize-overlap
    if inpf is None:
        stream = QueuedStreamBase(kind='input', blocksize=incframes, **kwargs)
        dtype = stream.dtype
        channels = stream.channels
    else:
        stream = SoundFileStream(inpf=inpf, blocksize=incframes, **kwargs)
        dtype = stream.dtype[0]
        channels = stream.channels[0]

    rxqueue = stream.rxq

    if channels > 1:
        always_2d = True

    outbuff = np.zeros((blocksize, channels) if always_2d else blocksize*channels, dtype=dtype)

    with stream:
        while stream.active:
            try:
                item = rxqueue.get(timeout=1)
            except queue.Empty:
                raise queue.Empty("Timed out waiting for data.")
            if item is None: break

            rxbuff = np.frombuffer(item, dtype=dtype)
            if always_2d:
                rxbuff.shape = (len(rxbuff)//channels, channels)
            outbuff[overlap:overlap+len(rxbuff)] = rxbuff

            yield outbuff[:] if copy else outbuff

            outbuff[:-incframes] = outbuff[incframes:]
