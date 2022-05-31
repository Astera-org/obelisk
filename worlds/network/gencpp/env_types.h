/**
 * Autogenerated by Thrift Compiler (0.16.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#ifndef env_TYPES_H
#define env_TYPES_H

#include <iosfwd>

#include <thrift/Thrift.h>
#include <thrift/TApplicationException.h>
#include <thrift/TBase.h>
#include <thrift/protocol/TProtocol.h>
#include <thrift/transport/TTransport.h>

#include <functional>
#include <memory>


namespace env {

typedef std::map<std::string, class SpaceSpec>  Space;

typedef std::map<std::string, class ETensor>  Observations;

typedef std::map<std::string, class Action>  Actions;

class Shape;

class ETensor;

class SpaceSpec;

class Action;

typedef struct _Shape__isset {
  _Shape__isset() : stride(false), names(false) {}
  bool stride :1;
  bool names :1;
} _Shape__isset;

class Shape : public virtual ::apache::thrift::TBase {
 public:

  Shape(const Shape&);
  Shape& operator=(const Shape&);
  Shape() noexcept {
  }

  virtual ~Shape() noexcept;
  std::vector<int32_t>  shape;
  std::vector<int32_t>  stride;
  std::vector<std::string>  names;

  _Shape__isset __isset;

  void __set_shape(const std::vector<int32_t> & val);

  void __set_stride(const std::vector<int32_t> & val);

  void __set_names(const std::vector<std::string> & val);

  bool operator == (const Shape & rhs) const
  {
    if (!(shape == rhs.shape))
      return false;
    if (__isset.stride != rhs.__isset.stride)
      return false;
    else if (__isset.stride && !(stride == rhs.stride))
      return false;
    if (__isset.names != rhs.__isset.names)
      return false;
    else if (__isset.names && !(names == rhs.names))
      return false;
    return true;
  }
  bool operator != (const Shape &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const Shape & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot) override;
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const override;

  virtual void printTo(std::ostream& out) const;
};

void swap(Shape &a, Shape &b);

std::ostream& operator<<(std::ostream& out, const Shape& obj);


class ETensor : public virtual ::apache::thrift::TBase {
 public:

  ETensor(const ETensor&);
  ETensor& operator=(const ETensor&);
  ETensor() noexcept {
  }

  virtual ~ETensor() noexcept;
  Shape shape;
  std::vector<double>  values;

  void __set_shape(const Shape& val);

  void __set_values(const std::vector<double> & val);

  bool operator == (const ETensor & rhs) const
  {
    if (!(shape == rhs.shape))
      return false;
    if (!(values == rhs.values))
      return false;
    return true;
  }
  bool operator != (const ETensor &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ETensor & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot) override;
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const override;

  virtual void printTo(std::ostream& out) const;
};

void swap(ETensor &a, ETensor &b);

std::ostream& operator<<(std::ostream& out, const ETensor& obj);

typedef struct _SpaceSpec__isset {
  _SpaceSpec__isset() : shape(false), min(false), max(false), discreteLabels(false) {}
  bool shape :1;
  bool min :1;
  bool max :1;
  bool discreteLabels :1;
} _SpaceSpec__isset;

class SpaceSpec : public virtual ::apache::thrift::TBase {
 public:

  SpaceSpec(const SpaceSpec&);
  SpaceSpec& operator=(const SpaceSpec&);
  SpaceSpec() noexcept
            : min(0),
              max(0) {
  }

  virtual ~SpaceSpec() noexcept;
  Shape shape;
  double min;
  double max;
  std::vector<std::string>  discreteLabels;

  _SpaceSpec__isset __isset;

  void __set_shape(const Shape& val);

  void __set_min(const double val);

  void __set_max(const double val);

  void __set_discreteLabels(const std::vector<std::string> & val);

  bool operator == (const SpaceSpec & rhs) const
  {
    if (__isset.shape != rhs.__isset.shape)
      return false;
    else if (__isset.shape && !(shape == rhs.shape))
      return false;
    if (!(min == rhs.min))
      return false;
    if (!(max == rhs.max))
      return false;
    if (__isset.discreteLabels != rhs.__isset.discreteLabels)
      return false;
    else if (__isset.discreteLabels && !(discreteLabels == rhs.discreteLabels))
      return false;
    return true;
  }
  bool operator != (const SpaceSpec &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const SpaceSpec & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot) override;
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const override;

  virtual void printTo(std::ostream& out) const;
};

void swap(SpaceSpec &a, SpaceSpec &b);

std::ostream& operator<<(std::ostream& out, const SpaceSpec& obj);

typedef struct _Action__isset {
  _Action__isset() : actionShape(false), vector(false), discreteOption(false) {}
  bool actionShape :1;
  bool vector :1;
  bool discreteOption :1;
} _Action__isset;

class Action : public virtual ::apache::thrift::TBase {
 public:

  Action(const Action&);
  Action& operator=(const Action&);
  Action() noexcept
         : discreteOption(0) {
  }

  virtual ~Action() noexcept;
  SpaceSpec actionShape;
  ETensor vector;
  int32_t discreteOption;

  _Action__isset __isset;

  void __set_actionShape(const SpaceSpec& val);

  void __set_vector(const ETensor& val);

  void __set_discreteOption(const int32_t val);

  bool operator == (const Action & rhs) const
  {
    if (__isset.actionShape != rhs.__isset.actionShape)
      return false;
    else if (__isset.actionShape && !(actionShape == rhs.actionShape))
      return false;
    if (__isset.vector != rhs.__isset.vector)
      return false;
    else if (__isset.vector && !(vector == rhs.vector))
      return false;
    if (!(discreteOption == rhs.discreteOption))
      return false;
    return true;
  }
  bool operator != (const Action &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const Action & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot) override;
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const override;

  virtual void printTo(std::ostream& out) const;
};

void swap(Action &a, Action &b);

std::ostream& operator<<(std::ostream& out, const Action& obj);

} // namespace

#endif
