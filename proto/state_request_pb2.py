# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: state_request.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='state_request.proto',
  package='prism',
  syntax='proto3',
  serialized_pb=_b('\n\x13state_request.proto\x12\x05prism\"\x1b\n\nStateFloat\x12\r\n\x05value\x18\x01 \x03(\x01\"\x1f\n\x0eStringVarNames\x12\r\n\x05value\x18\x01 \x03(\tb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_STATEFLOAT = _descriptor.Descriptor(
  name='StateFloat',
  full_name='prism.StateFloat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='prism.StateFloat.value', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=57,
)


_STRINGVARNAMES = _descriptor.Descriptor(
  name='StringVarNames',
  full_name='prism.StringVarNames',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='prism.StringVarNames.value', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=90,
)

DESCRIPTOR.message_types_by_name['StateFloat'] = _STATEFLOAT
DESCRIPTOR.message_types_by_name['StringVarNames'] = _STRINGVARNAMES

StateFloat = _reflection.GeneratedProtocolMessageType('StateFloat', (_message.Message,), dict(
  DESCRIPTOR = _STATEFLOAT,
  __module__ = 'state_request_pb2'
  # @@protoc_insertion_point(class_scope:prism.StateFloat)
  ))
_sym_db.RegisterMessage(StateFloat)

StringVarNames = _reflection.GeneratedProtocolMessageType('StringVarNames', (_message.Message,), dict(
  DESCRIPTOR = _STRINGVARNAMES,
  __module__ = 'state_request_pb2'
  # @@protoc_insertion_point(class_scope:prism.StringVarNames)
  ))
_sym_db.RegisterMessage(StringVarNames)


# @@protoc_insertion_point(module_scope)
