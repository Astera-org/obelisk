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
  this.agentID = null;
  this.worldID = null;
  this.agentCfg = null;
  this.worldCfg = null;
  if (args) {
    if (args.jobID !== undefined && args.jobID !== null) {
      this.jobID = args.jobID;
    }
    if (args.agentID !== undefined && args.agentID !== null) {
      this.agentID = args.agentID;
    }
    if (args.worldID !== undefined && args.worldID !== null) {
      this.worldID = args.worldID;
    }
    if (args.agentCfg !== undefined && args.agentCfg !== null) {
      this.agentCfg = args.agentCfg;
    }
    if (args.worldCfg !== undefined && args.worldCfg !== null) {
      this.worldCfg = args.worldCfg;
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
      if (ftype == Thrift.Type.I32) {
        this.agentID = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.I32) {
        this.worldID = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.STRING) {
        this.agentCfg = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.STRING) {
        this.worldCfg = input.readString().value;
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
  if (this.agentID !== null && this.agentID !== undefined) {
    output.writeFieldBegin('agentID', Thrift.Type.I32, 2);
    output.writeI32(this.agentID);
    output.writeFieldEnd();
  }
  if (this.worldID !== null && this.worldID !== undefined) {
    output.writeFieldBegin('worldID', Thrift.Type.I32, 3);
    output.writeI32(this.worldID);
    output.writeFieldEnd();
  }
  if (this.agentCfg !== null && this.agentCfg !== undefined) {
    output.writeFieldBegin('agentCfg', Thrift.Type.STRING, 4);
    output.writeString(this.agentCfg);
    output.writeFieldEnd();
  }
  if (this.worldCfg !== null && this.worldCfg !== undefined) {
    output.writeFieldBegin('worldCfg', Thrift.Type.STRING, 5);
    output.writeString(this.worldCfg);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

ResultWork = function(args) {
  this.jobID = null;
  this.status = null;
  this.cycles = null;
  this.seconds = null;
  this.score = null;
  this.workerName = null;
  this.instanceName = null;
  if (args) {
    if (args.jobID !== undefined && args.jobID !== null) {
      this.jobID = args.jobID;
    }
    if (args.status !== undefined && args.status !== null) {
      this.status = args.status;
    }
    if (args.cycles !== undefined && args.cycles !== null) {
      this.cycles = args.cycles;
    }
    if (args.seconds !== undefined && args.seconds !== null) {
      this.seconds = args.seconds;
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
        this.status = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.I32) {
        this.cycles = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.I32) {
        this.seconds = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 6:
      if (ftype == Thrift.Type.DOUBLE) {
        this.score = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 7:
      if (ftype == Thrift.Type.STRING) {
        this.workerName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 8:
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
  if (this.status !== null && this.status !== undefined) {
    output.writeFieldBegin('status', Thrift.Type.I32, 2);
    output.writeI32(this.status);
    output.writeFieldEnd();
  }
  if (this.cycles !== null && this.cycles !== undefined) {
    output.writeFieldBegin('cycles', Thrift.Type.I32, 3);
    output.writeI32(this.cycles);
    output.writeFieldEnd();
  }
  if (this.seconds !== null && this.seconds !== undefined) {
    output.writeFieldBegin('seconds', Thrift.Type.I32, 4);
    output.writeI32(this.seconds);
    output.writeFieldEnd();
  }
  if (this.score !== null && this.score !== undefined) {
    output.writeFieldBegin('score', Thrift.Type.DOUBLE, 6);
    output.writeDouble(this.score);
    output.writeFieldEnd();
  }
  if (this.workerName !== null && this.workerName !== undefined) {
    output.writeFieldBegin('workerName', Thrift.Type.STRING, 7);
    output.writeString(this.workerName);
    output.writeFieldEnd();
  }
  if (this.instanceName !== null && this.instanceName !== undefined) {
    output.writeFieldBegin('instanceName', Thrift.Type.STRING, 8);
    output.writeString(this.instanceName);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

BinInfo = function(args) {
  this.binID = null;
  this.name = null;
  this.version = null;
  this.hash = null;
  if (args) {
    if (args.binID !== undefined && args.binID !== null) {
      this.binID = args.binID;
    }
    if (args.name !== undefined && args.name !== null) {
      this.name = args.name;
    }
    if (args.version !== undefined && args.version !== null) {
      this.version = args.version;
    }
    if (args.hash !== undefined && args.hash !== null) {
      this.hash = args.hash;
    }
  }
};
BinInfo.prototype = {};
BinInfo.prototype.read = function(input) {
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
        this.binID = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.STRING) {
        this.name = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.STRING) {
        this.version = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.STRING) {
        this.hash = input.readString().value;
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

BinInfo.prototype.write = function(output) {
  output.writeStructBegin('BinInfo');
  if (this.binID !== null && this.binID !== undefined) {
    output.writeFieldBegin('binID', Thrift.Type.I32, 1);
    output.writeI32(this.binID);
    output.writeFieldEnd();
  }
  if (this.name !== null && this.name !== undefined) {
    output.writeFieldBegin('name', Thrift.Type.STRING, 2);
    output.writeString(this.name);
    output.writeFieldEnd();
  }
  if (this.version !== null && this.version !== undefined) {
    output.writeFieldBegin('version', Thrift.Type.STRING, 3);
    output.writeString(this.version);
    output.writeFieldEnd();
  }
  if (this.hash !== null && this.hash !== undefined) {
    output.writeFieldBegin('hash', Thrift.Type.STRING, 4);
    output.writeString(this.hash);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

