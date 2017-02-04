"""
Loopback tests for pastream.
"""
from __future__ import print_function
import os, sys
import numpy as np
import soundfile as sf
import pytest
import numpy.testing as npt
import time
import tempfile
try:
    import Queue as queue
except ImportError:
    import queue
import utils

BLOCKSIZE = 2048
TEST_LENGTHS = [5]
PREAMBLE = 0x7FFFFFFF

vhex = np.vectorize('{:#10x}'.format)
tohex = lambda x: vhex(x.view('u4'))


def sf_find_delay(xf, mask=0xFFFF0000, chan=None):
    pos = xf.tell()

    off = -1
    inpblocks = xf.blocks(BLOCKSIZE, dtype='int32')
    for i,inpblk in enumerate(inpblocks):
        nonzeros = np.where(inpblk[:, chan]&mask)
        if nonzeros[0].any(): 
            off = i*BLOCKSIZE + nonzeros[0][0]
            break

    xf.seek(pos)

    return off

def sf_assert_equal(tx_fh, rx_fh, mask=0xFFFF0000, chi=None, chf=None, allow_truncation=False, allow_delay=False):
    d = sf_find_delay(rx_fh, mask=mask, chan=chi)
    
    assert d != -1, "Test Preamble pattern not found"

    if not allow_delay:
        assert d == 0, "Signal delayed by %d frames" % d
    rx_fh.seek(d)
    
    inpblocks = tx_fh.blocks(BLOCKSIZE, dtype='int32')
    for inpblk in inpblocks:
        outblk = rx_fh.read(BLOCKSIZE, dtype='int32')

        if not allow_truncation:
            assert len(inpblk) == len(outblk), "Some samples were dropped"

        mlen = min(len(inpblk), len(outblk))
        inp = (inpblk[:mlen,chi:chf].view('u4'))&mask
        out = outblk[:mlen,chi:chf].view('u4')&mask

        npt.assert_array_equal(inp, out, "Loopback data mismatch")


class PortAudioLoopbackTester(object):
    def _gen_random(self, rdm_fh, nrepeats, nbytes):
        shift = 8*(4-nbytes)
        minval = -(0x80000000>>shift)
        maxval = 0x7FFFFFFF>>shift

        preamble = np.zeros((rdm_fh.samplerate//10, rdm_fh.channels), dtype=np.int32)
        preamble[:] = maxval << shift
        rdm_fh.write(preamble)

        for i in range(nrepeats):
            pattern = np.random.randint(minval, maxval+1, (rdm_fh.samplerate, rdm_fh.channels)) << shift
            rdm_fh.write(pattern.astype(np.int32))

    @pytest.fixture(scope='session', params=TEST_LENGTHS)
    def randomwav84832(self, request, tmpdir_factory):
        tmpdir = tmpdir_factory.getbasetemp()
        rdmf = tempfile.NamedTemporaryFile('w+b', dir=str(tmpdir))

        rdm_fh = sf.SoundFile(rdmf, 'w+', 48000, 8, 'PCM_32', format='wav')
        self._gen_random(rdm_fh, request.param, 4)
        rdm_fh.seek(0)

        yield rdm_fh

        rdm_fh.close()

    @pytest.fixture(scope='session', params=TEST_LENGTHS)
    def randomwav84824(self, request, tmpdir_factory):
        tmpdir = tmpdir_factory.getbasetemp()
        rdmf = tempfile.NamedTemporaryFile('w+b', dir=str(tmpdir))

        rdm_fh = sf.SoundFile(rdmf, 'w+', 48000, 8, 'PCM_24', format='wav')
        self._gen_random(rdm_fh, request.param, 3)
        rdm_fh.seek(0)

        yield rdm_fh

        rdm_fh.close()

    @pytest.fixture(scope='session', params=TEST_LENGTHS)
    def randomwav84816(self, request, tmpdir_factory):
        tmpdir = tmpdir_factory.getbasetemp()
        rdmf = tempfile.NamedTemporaryFile('w+b', dir=str(tmpdir))

        rdm_fh = sf.SoundFile(rdmf, 'w+', 48000, 8, 'PCM_16', format='wav')
        self._gen_random(rdm_fh, request.param, 2)
        rdm_fh.seek(0)

        yield rdm_fh

        rdm_fh.close()

    @pytest.fixture(scope='session', params=TEST_LENGTHS)
    def randomwav84432(self, request, tmpdir_factory):
        tmpdir = tmpdir_factory.getbasetemp()
        rdmf = tempfile.NamedTemporaryFile('w+b', dir=str(tmpdir))

        rdm_fh = sf.SoundFile(rdmf, 'w+', 44100, 8, 'PCM_32', format='wav')
        self._gen_random(rdm_fh, request.param, 4)
        rdm_fh.seek(0)

        yield rdm_fh

        rdm_fh.close()

    @pytest.fixture(scope='session', params=TEST_LENGTHS)
    def randomwav84416(self, request, tmpdir_factory):
        tmpdir = tmpdir_factory.getbasetemp()
        rdmf = tempfile.NamedTemporaryFile('w+b', dir=str(tmpdir))

        rdm_fh = sf.SoundFile(rdmf, 'w+', 44100, 8, 'PCM_16', format='wav')
        self._gen_random(rdm_fh, request.param, 2)
        rdm_fh.seek(0)

        yield rdm_fh

        rdm_fh.close()

    def assert_stream_equal(self, inp_fh, preamble, **kwargs):
        inpf2 = sf.SoundFile(inp_fh.name.name, mode='rb')    

        delay = -1
        found_delay = False
        nframes = mframes = 0
        for outframes in utils.blockstream(inp_fh, **kwargs):
            if not found_delay:
                matches = outframes[:, 0].view('u4') == preamble
                if np.any(matches): 
                    found_delay = True
                    nonzeros = np.where(matches)[0]
                    outframes = outframes[nonzeros[0]:]
                    nframes += nonzeros[0]
                    delay = nframes
            if found_delay:
                inframes = inpf2.read(len(outframes), dtype='int32', always_2d=True)

                mlen = min(len(inframes), len(outframes))
                inp = inframes[:mlen].view('u4')
                out = outframes[:mlen].view('u4')

                npt.assert_array_equal(inp, out, "Loopback data mismatch")
                mframes += mlen
            nframes += len(outframes)

        assert delay != -1, "Preamble not found on loopback"

        print("Matched %d of %d frames; Initial delay of %d frames" % (mframes, nframes, delay))

class TestALSALoopback(PortAudioLoopbackTester):
    def test_wav32(self, tmpdir, randomwav84832):
        tx_fh = randomwav84832
        tx_fh.seek(0)
        self.assert_stream_equal(tx_fh, PREAMBLE,
                                 blocksize=512, dtype='int32',
                                 device='aduplex')
