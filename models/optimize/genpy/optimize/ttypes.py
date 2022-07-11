#
# Autogenerated by Thrift Compiler (0.16.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TFrozenDict, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

import sys

from thrift.transport import TTransport
all_structs = []


class HyperParameter(object):
    """
    Attributes:
     - name
     - center
     - stddev
     - min
     - max
     - type_of_scale
     - is_integer

    """


    def __init__(self, name=None, center=None, stddev=None, min=None, max=None, type_of_scale=None, is_integer=None,):
        self.name = name
        self.center = center
        self.stddev = stddev
        self.min = min
        self.max = max
        self.type_of_scale = type_of_scale
        self.is_integer = is_integer

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.name = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.DOUBLE:
                    self.center = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.DOUBLE:
                    self.stddev = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.DOUBLE:
                    self.min = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.DOUBLE:
                    self.max = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 6:
                if ftype == TType.STRING:
                    self.type_of_scale = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 7:
                if ftype == TType.BOOL:
                    self.is_integer = iprot.readBool()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('HyperParameter')
        if self.name is not None:
            oprot.writeFieldBegin('name', TType.STRING, 1)
            oprot.writeString(self.name.encode('utf-8') if sys.version_info[0] == 2 else self.name)
            oprot.writeFieldEnd()
        if self.center is not None:
            oprot.writeFieldBegin('center', TType.DOUBLE, 2)
            oprot.writeDouble(self.center)
            oprot.writeFieldEnd()
        if self.stddev is not None:
            oprot.writeFieldBegin('stddev', TType.DOUBLE, 3)
            oprot.writeDouble(self.stddev)
            oprot.writeFieldEnd()
        if self.min is not None:
            oprot.writeFieldBegin('min', TType.DOUBLE, 4)
            oprot.writeDouble(self.min)
            oprot.writeFieldEnd()
        if self.max is not None:
            oprot.writeFieldBegin('max', TType.DOUBLE, 5)
            oprot.writeDouble(self.max)
            oprot.writeFieldEnd()
        if self.type_of_scale is not None:
            oprot.writeFieldBegin('type_of_scale', TType.STRING, 6)
            oprot.writeString(self.type_of_scale.encode('utf-8') if sys.version_info[0] == 2 else self.type_of_scale)
            oprot.writeFieldEnd()
        if self.is_integer is not None:
            oprot.writeFieldBegin('is_integer', TType.BOOL, 7)
            oprot.writeBool(self.is_integer)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.name is None:
            raise TProtocolException(message='Required field name is unset!')
        if self.center is None:
            raise TProtocolException(message='Required field center is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class Suggestions(object):
    """
    Attributes:
     - observationId
     - parameterSuggestions
     - metadata

    """


    def __init__(self, observationId=None, parameterSuggestions=None, metadata=None,):
        self.observationId = observationId
        self.parameterSuggestions = parameterSuggestions
        self.metadata = metadata

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.observationId = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.MAP:
                    self.parameterSuggestions = {}
                    (_ktype1, _vtype2, _size0) = iprot.readMapBegin()
                    for _i4 in range(_size0):
                        _key5 = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                        _val6 = iprot.readDouble()
                        self.parameterSuggestions[_key5] = _val6
                    iprot.readMapEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.MAP:
                    self.metadata = {}
                    (_ktype8, _vtype9, _size7) = iprot.readMapBegin()
                    for _i11 in range(_size7):
                        _key12 = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                        _val13 = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                        self.metadata[_key12] = _val13
                    iprot.readMapEnd()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('Suggestions')
        if self.observationId is not None:
            oprot.writeFieldBegin('observationId', TType.STRING, 1)
            oprot.writeString(self.observationId.encode('utf-8') if sys.version_info[0] == 2 else self.observationId)
            oprot.writeFieldEnd()
        if self.parameterSuggestions is not None:
            oprot.writeFieldBegin('parameterSuggestions', TType.MAP, 2)
            oprot.writeMapBegin(TType.STRING, TType.DOUBLE, len(self.parameterSuggestions))
            for kiter14, viter15 in self.parameterSuggestions.items():
                oprot.writeString(kiter14.encode('utf-8') if sys.version_info[0] == 2 else kiter14)
                oprot.writeDouble(viter15)
            oprot.writeMapEnd()
            oprot.writeFieldEnd()
        if self.metadata is not None:
            oprot.writeFieldBegin('metadata', TType.MAP, 3)
            oprot.writeMapBegin(TType.STRING, TType.STRING, len(self.metadata))
            for kiter16, viter17 in self.metadata.items():
                oprot.writeString(kiter16.encode('utf-8') if sys.version_info[0] == 2 else kiter16)
                oprot.writeString(viter17.encode('utf-8') if sys.version_info[0] == 2 else viter17)
            oprot.writeMapEnd()
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(HyperParameter)
HyperParameter.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'name', 'UTF8', None, ),  # 1
    (2, TType.DOUBLE, 'center', None, None, ),  # 2
    (3, TType.DOUBLE, 'stddev', None, None, ),  # 3
    (4, TType.DOUBLE, 'min', None, None, ),  # 4
    (5, TType.DOUBLE, 'max', None, None, ),  # 5
    (6, TType.STRING, 'type_of_scale', 'UTF8', None, ),  # 6
    (7, TType.BOOL, 'is_integer', None, None, ),  # 7
)
all_structs.append(Suggestions)
Suggestions.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'observationId', 'UTF8', None, ),  # 1
    (2, TType.MAP, 'parameterSuggestions', (TType.STRING, 'UTF8', TType.DOUBLE, None, False), None, ),  # 2
    (3, TType.MAP, 'metadata', (TType.STRING, 'UTF8', TType.STRING, 'UTF8', False), None, ),  # 3
)
fix_spec(all_structs)
del all_structs
