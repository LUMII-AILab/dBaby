#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2016, 2017 Didzis Gosko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import struct, io
import numpy as np

try:
    # Python 3, use pure Python pickle implementation
    from pickle import _Pickler, _Unpickler
except:
    # Python 2.7
    from pickle import Pickler as _Pickler, Unpickler as _Unpickler

try:
    # Python 2.7
    buffer
except NameError:
    # Python 3
    buffer = memoryview

NUMPY_NDARRAY_OPCODE = b'n'


class Pickler(_Pickler):

    def __init__(self, file, protocol=2, use_tofile=True):
        if protocol != 2 and protocol != 3:
            raise ValueError("npickle supports only protocols 2 and 3 (Python 3)")
        _Pickler.__init__(self, file, protocol)
        self.use_tofile = use_tofile
        self.file = file
        self.dispatch[np.ndarray] = Pickler.save_numpy_ndarray

    def save_numpy_ndarray(self, obj, pack=struct.pack):
        self.write(NUMPY_NDARRAY_OPCODE)            # opcode for numpy arrays
        dtype = str(obj.dtype).encode('utf8')       # prepare type string
        self.write(pack('B', len(dtype)))           # write type string size
        self.write(dtype)                           # write type string
        self.write(pack('I', obj.ndim))             # write number of dimensions
        self.write(pack('I'*obj.ndim, *obj.shape))  # write shape
        if self.use_tofile:
            try:
                obj.tofile(self.file)               # write numpy array data with tofile
            except AttributeError:
                self.use_tofile = False             # fallback
            except io.UnsupportedOperation:
                self.use_tofile = False             # fallback
        if not self.use_tofile:
            if buffer is memoryview:                # Python 3
                if obj.data.c_contiguous:
                    self.write(obj.data)            # write contiguous buffer directly
                else:
                    self.write(bytes(obj.data))     # inefficient, first make a contiguous copy, then write to file
            else:                                   # Python 2.7
                try:
                    self.write(obj.data)            # write contiguous buffer directly
                except AttributeError:
                    self.write(np.copy(obj).data)   # inefficient, first make (a contiguous) copy, then write to file
        self.memoize(obj)


class Unpickler(_Unpickler):

    def __init__(self, file):
        _Unpickler.__init__(self, file)
        self.file = file
        self.dispatch[NUMPY_NDARRAY_OPCODE[0]] = Unpickler.load_numpy_ndarray

    def load_numpy_ndarray(self):
        dtype = np.dtype(self.read(struct.unpack('B', self.read(1))[0]))
        ndim = struct.unpack('I', self.read(4))[0]
        shape = struct.unpack('I'*ndim, self.read(ndim*4))
        # number of elements in array
        count = 1
        for sz in shape:
            count *= sz
        # NOTE: in Python 3 numpy's fromfile() consumes all file in case the file is a pipe,
        #       better manually read the data from file and let numpy reinterpret the data as ndarray
        array = np.frombuffer(buffer(self.file.read(dtype.itemsize*count)), dtype=dtype, count=count)
        # array = np.fromfile(self.file, dtype=dtype, count=count, sep='')  # legacy
        array = array.reshape(shape)                # restore original shape
        self.append(array)


def dump(obj, filename):
    with open(filename, 'wb') as f:
        Pickler(f).dump(obj)

def load(filename):
    with open(filename, 'rb') as f:
        return Unpickler(f).load()


# Convenience functions for compressed pickle output (in worst case has to make copies or use external utilities and pipes)
# NOTE: because of using numpy's tofile (and previously fromfile), pseudo file-like Python objects are not supported (like GzipFile)

def dump_bzip2(obj, filename, use_pipe=False):
    if not use_pipe:
        # can be the most inefficient in worst case (when storing np.ndarray slices), but the most compatible
        import bz2
        try:
            _open = bz2.open
        except AttributeError:
            _open = bz2.BZ2File
        with _open(filename, 'wb') as f:
            Pickler(f, use_tofile=False).dump(obj)
    else:
        # will not work if bzip2 is not available
        import os, pipes
        t = pipes.Template()
        t.append('bzip2 --compress', '--')
        with t.open(filename, 'w') as f:
            try:
                f = os.fdopen(f.fileno(), 'wb', buffering=0)
            except TypeError:
                pass
            Pickler(f).dump(obj)

def load_bzip2(filename, use_pipe=False):
    if not use_pipe:
        # the most compatible
        import bz2
        try:
            _open = bz2.open
        except AttributeError:
            _open = bz2.BZ2File
        with _open(filename, 'rb') as f:
            return Unpickler(f).load()
    else:
        # legacy code using pipes, will not work if bzip2 is not available
        import os, pipes
        t = pipes.Template()
        t.append('bzip2 --decompress --stdout', '--')
        with t.open(filename, 'r') as f:
            try:
                f = os.fdopen(f.fileno(), 'rb', buffering=0)
            except TypeError:
                pass
            return Unpickler(f).load()

def dump_gzip(obj, filename, use_pipe=False):
    if not use_pipe:
        # can be the most inefficient in worst case (when storing np.ndarray slices), but the most compatible
        import gzip
        try:
            _open = gzip.open
        except AttributeError:
            _open = gzip.GzipFile
        with _open(filename, 'wb') as f:
            Pickler(f, use_tofile=False).dump(obj)
    else:
        # will not work if gzip is not available
        import os, pipes
        t = pipes.Template()
        t.append('gzip', '--')
        with t.open(filename, 'w') as f:
            try:
                f = os.fdopen(f.fileno(), 'wb', buffering=0)
            except TypeError:
                pass
            Pickler(f).dump(obj)

def load_gzip(filename, use_pipe=False):
    if not use_pipe:
        # the most compatible
        import gzip
        try:
            _open = gzip.open
        except AttributeError:
            _open = gzip.GzipFile
        with _open(filename, 'rb') as f:
            return Unpickler(f).load()
    else:
        # legacy code using pipes, will not work if gzip is not available
        import os, pipes
        t = pipes.Template()
        t.append('gzip --decompress --stdout', '--')
        with t.open(filename, 'r') as f:
            try:
                f = os.fdopen(f.fileno(), 'rb', buffering=0)
            except TypeError:
                pass
            return Unpickler(f).load()

