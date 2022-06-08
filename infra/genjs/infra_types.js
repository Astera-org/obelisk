//
// Autogenerated by Thrift Compiler (0.16.0)
//
// DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
//
if (typeof Int64 === 'undefined' && typeof require === 'function') {
  var Int64 = require('node-int64');
}


Job = function(args) {
  this.jobID = null;
  this.agentName = null;
  this.worldName = null;
  if (args) {
    if (args.jobID !== undefined && args.jobID !== null) {
      this.jobID = args.jobID;
    }
    if (args.agentName !== undefined && args.agentName !== null) {
      this.agentName = args.agentName;
    }
    if (args.worldName !== undefined && args.worldName !== null) {
      this.worldName = args.worldName;
    }
  }
};
Job.prototype = {};
Job.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.I32) {
        this.jobID = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.STRING) {
        this.agentName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.STRING) {
        this.worldName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Job.prototype.write = function(output) {
  output.writeStructBegin('Job');
  if (this.jobID !== null && this.jobID !== undefined) {
    output.writeFieldBegin('jobID', Thrift.Type.I32, 1);
    output.writeI32(this.jobID);
    output.writeFieldEnd();
  }
  if (this.agentName !== null && this.agentName !== undefined) {
    output.writeFieldBegin('agentName', Thrift.Type.STRING, 2);
    output.writeString(this.agentName);
    output.writeFieldEnd();
  }
  if (this.worldName !== null && this.worldName !== undefined) {
    output.writeFieldBegin('worldName', Thrift.Type.STRING, 3);
    output.writeString(this.worldName);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

ResultWork = function(args) {
  this.jobID = null;
  this.cycles = null;
  this.timeStart = null;
  this.timeStop = null;
  this.score = null;
  this.workerName = null;
  this.instanceName = null;
  if (args) {
    if (args.jobID !== undefined && args.jobID !== null) {
      this.jobID = args.jobID;
    }
    if (args.cycles !== undefined && args.cycles !== null) {
      this.cycles = args.cycles;
    }
    if (args.timeStart !== undefined && args.timeStart !== null) {
      this.timeStart = args.timeStart;
    }
    if (args.timeStop !== undefined && args.timeStop !== null) {
      this.timeStop = args.timeStop;
    }
    if (args.score !== undefined && args.score !== null) {
      this.score = args.score;
    }
    if (args.workerName !== undefined && args.workerName !== null) {
      this.workerName = args.workerName;
    }
    if (args.instanceName !== undefined && args.instanceName !== null) {
      this.instanceName = args.instanceName;
    }
  }
};
ResultWork.prototype = {};
ResultWork.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.I32) {
        this.jobID = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.I32) {
        this.cycles = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.I32) {
        this.timeStart = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.I32) {
        this.timeStop = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.DOUBLE) {
        this.score = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 6:
      if (ftype == Thrift.Type.STRING) {
        this.workerName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 7:
      if (ftype == Thrift.Type.STRING) {
        this.instanceName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

ResultWork.prototype.write = function(output) {
  output.writeStructBegin('ResultWork');
  if (this.jobID !== null && this.jobID !== undefined) {
    output.writeFieldBegin('jobID', Thrift.Type.I32, 1);
    output.writeI32(this.jobID);
    output.writeFieldEnd();
  }
  if (this.cycles !== null && this.cycles !== undefined) {
    output.writeFieldBegin('cycles', Thrift.Type.I32, 2);
    output.writeI32(this.cycles);
    output.writeFieldEnd();
  }
  if (this.timeStart !== null && this.timeStart !== undefined) {
    output.writeFieldBegin('timeStart', Thrift.Type.I32, 3);
    output.writeI32(this.timeStart);
    output.writeFieldEnd();
  }
  if (this.timeStop !== null && this.timeStop !== undefined) {
    output.writeFieldBegin('timeStop', Thrift.Type.I32, 4);
    output.writeI32(this.timeStop);
    output.writeFieldEnd();
  }
  if (this.score !== null && this.score !== undefined) {
    output.writeFieldBegin('score', Thrift.Type.DOUBLE, 5);
    output.writeDouble(this.score);
    output.writeFieldEnd();
  }
  if (this.workerName !== null && this.workerName !== undefined) {
    output.writeFieldBegin('workerName', Thrift.Type.STRING, 6);
    output.writeString(this.workerName);
    output.writeFieldEnd();
  }
  if (this.instanceName !== null && this.instanceName !== undefined) {
    output.writeFieldBegin('instanceName', Thrift.Type.STRING, 7);
    output.writeString(this.instanceName);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};
