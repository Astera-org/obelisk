//
// Autogenerated by Thrift Compiler (0.16.0)
//
// DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
//
if (typeof Int64 === 'undefined' && typeof require === 'function') {
  var Int64 = require('node-int64');
}


Job = function(args) {
  this.job_id = null;
  this.agent_id = null;
  this.world_id = null;
  this.agent_param = null;
  this.world_param = null;
  if (args) {
    if (args.job_id !== undefined && args.job_id !== null) {
      this.job_id = args.job_id;
    }
    if (args.agent_id !== undefined && args.agent_id !== null) {
      this.agent_id = args.agent_id;
    }
    if (args.world_id !== undefined && args.world_id !== null) {
      this.world_id = args.world_id;
    }
    if (args.agent_param !== undefined && args.agent_param !== null) {
      this.agent_param = args.agent_param;
    }
    if (args.world_param !== undefined && args.world_param !== null) {
      this.world_param = args.world_param;
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
        this.job_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.I32) {
        this.agent_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.I32) {
        this.world_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.STRING) {
        this.agent_param = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.STRING) {
        this.world_param = input.readString().value;
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
  if (this.job_id !== null && this.job_id !== undefined) {
    output.writeFieldBegin('job_id', Thrift.Type.I32, 1);
    output.writeI32(this.job_id);
    output.writeFieldEnd();
  }
  if (this.agent_id !== null && this.agent_id !== undefined) {
    output.writeFieldBegin('agent_id', Thrift.Type.I32, 2);
    output.writeI32(this.agent_id);
    output.writeFieldEnd();
  }
  if (this.world_id !== null && this.world_id !== undefined) {
    output.writeFieldBegin('world_id', Thrift.Type.I32, 3);
    output.writeI32(this.world_id);
    output.writeFieldEnd();
  }
  if (this.agent_param !== null && this.agent_param !== undefined) {
    output.writeFieldBegin('agent_param', Thrift.Type.STRING, 4);
    output.writeString(this.agent_param);
    output.writeFieldEnd();
  }
  if (this.world_param !== null && this.world_param !== undefined) {
    output.writeFieldBegin('world_param', Thrift.Type.STRING, 5);
    output.writeString(this.world_param);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

ResultJob = function(args) {
  this.job_id = null;
  this.status = null;
  this.seconds = null;
  this.steps = null;
  this.targetSteps = null;
  this.cycles = null;
  this.score = null;
  this.worker_name = null;
  this.instance_id = null;
  if (args) {
    if (args.job_id !== undefined && args.job_id !== null) {
      this.job_id = args.job_id;
    }
    if (args.status !== undefined && args.status !== null) {
      this.status = args.status;
    }
    if (args.seconds !== undefined && args.seconds !== null) {
      this.seconds = args.seconds;
    }
    if (args.steps !== undefined && args.steps !== null) {
      this.steps = args.steps;
    }
    if (args.targetSteps !== undefined && args.targetSteps !== null) {
      this.targetSteps = args.targetSteps;
    }
    if (args.cycles !== undefined && args.cycles !== null) {
      this.cycles = args.cycles;
    }
    if (args.score !== undefined && args.score !== null) {
      this.score = args.score;
    }
    if (args.worker_name !== undefined && args.worker_name !== null) {
      this.worker_name = args.worker_name;
    }
    if (args.instance_id !== undefined && args.instance_id !== null) {
      this.instance_id = args.instance_id;
    }
  }
};
ResultJob.prototype = {};
ResultJob.prototype.read = function(input) {
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
        this.job_id = input.readI32().value;
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
        this.seconds = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.I32) {
        this.steps = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.I32) {
        this.targetSteps = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 6:
      if (ftype == Thrift.Type.I32) {
        this.cycles = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 7:
      if (ftype == Thrift.Type.DOUBLE) {
        this.score = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 8:
      if (ftype == Thrift.Type.STRING) {
        this.worker_name = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 9:
      if (ftype == Thrift.Type.I32) {
        this.instance_id = input.readI32().value;
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

ResultJob.prototype.write = function(output) {
  output.writeStructBegin('ResultJob');
  if (this.job_id !== null && this.job_id !== undefined) {
    output.writeFieldBegin('job_id', Thrift.Type.I32, 1);
    output.writeI32(this.job_id);
    output.writeFieldEnd();
  }
  if (this.status !== null && this.status !== undefined) {
    output.writeFieldBegin('status', Thrift.Type.I32, 2);
    output.writeI32(this.status);
    output.writeFieldEnd();
  }
  if (this.seconds !== null && this.seconds !== undefined) {
    output.writeFieldBegin('seconds', Thrift.Type.I32, 3);
    output.writeI32(this.seconds);
    output.writeFieldEnd();
  }
  if (this.steps !== null && this.steps !== undefined) {
    output.writeFieldBegin('steps', Thrift.Type.I32, 4);
    output.writeI32(this.steps);
    output.writeFieldEnd();
  }
  if (this.targetSteps !== null && this.targetSteps !== undefined) {
    output.writeFieldBegin('targetSteps', Thrift.Type.I32, 5);
    output.writeI32(this.targetSteps);
    output.writeFieldEnd();
  }
  if (this.cycles !== null && this.cycles !== undefined) {
    output.writeFieldBegin('cycles', Thrift.Type.I32, 6);
    output.writeI32(this.cycles);
    output.writeFieldEnd();
  }
  if (this.score !== null && this.score !== undefined) {
    output.writeFieldBegin('score', Thrift.Type.DOUBLE, 7);
    output.writeDouble(this.score);
    output.writeFieldEnd();
  }
  if (this.worker_name !== null && this.worker_name !== undefined) {
    output.writeFieldBegin('worker_name', Thrift.Type.STRING, 8);
    output.writeString(this.worker_name);
    output.writeFieldEnd();
  }
  if (this.instance_id !== null && this.instance_id !== undefined) {
    output.writeFieldBegin('instance_id', Thrift.Type.I32, 9);
    output.writeI32(this.instance_id);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobInfo = function(args) {
  this.job_id = null;
  this.user_id = null;
  this.search_id = null;
  this.status = null;
  this.priority = null;
  this.callback = null;
  this.time_added = null;
  this.agent_id = null;
  this.world_id = null;
  this.agent_param = null;
  this.world_param = null;
  this.note = null;
  this.bail_threshold = null;
  this.worker_name = null;
  this.instance_id = null;
  this.worker_ip = null;
  this.time_handed = null;
  this.seconds = null;
  this.steps = null;
  this.cycles = null;
  this.bailed = null;
  this.score = null;
  if (args) {
    if (args.job_id !== undefined && args.job_id !== null) {
      this.job_id = args.job_id;
    }
    if (args.user_id !== undefined && args.user_id !== null) {
      this.user_id = args.user_id;
    }
    if (args.search_id !== undefined && args.search_id !== null) {
      this.search_id = args.search_id;
    }
    if (args.status !== undefined && args.status !== null) {
      this.status = args.status;
    }
    if (args.priority !== undefined && args.priority !== null) {
      this.priority = args.priority;
    }
    if (args.callback !== undefined && args.callback !== null) {
      this.callback = args.callback;
    }
    if (args.time_added !== undefined && args.time_added !== null) {
      this.time_added = args.time_added;
    }
    if (args.agent_id !== undefined && args.agent_id !== null) {
      this.agent_id = args.agent_id;
    }
    if (args.world_id !== undefined && args.world_id !== null) {
      this.world_id = args.world_id;
    }
    if (args.agent_param !== undefined && args.agent_param !== null) {
      this.agent_param = args.agent_param;
    }
    if (args.world_param !== undefined && args.world_param !== null) {
      this.world_param = args.world_param;
    }
    if (args.note !== undefined && args.note !== null) {
      this.note = args.note;
    }
    if (args.bail_threshold !== undefined && args.bail_threshold !== null) {
      this.bail_threshold = args.bail_threshold;
    }
    if (args.worker_name !== undefined && args.worker_name !== null) {
      this.worker_name = args.worker_name;
    }
    if (args.instance_id !== undefined && args.instance_id !== null) {
      this.instance_id = args.instance_id;
    }
    if (args.worker_ip !== undefined && args.worker_ip !== null) {
      this.worker_ip = args.worker_ip;
    }
    if (args.time_handed !== undefined && args.time_handed !== null) {
      this.time_handed = args.time_handed;
    }
    if (args.seconds !== undefined && args.seconds !== null) {
      this.seconds = args.seconds;
    }
    if (args.steps !== undefined && args.steps !== null) {
      this.steps = args.steps;
    }
    if (args.cycles !== undefined && args.cycles !== null) {
      this.cycles = args.cycles;
    }
    if (args.bailed !== undefined && args.bailed !== null) {
      this.bailed = args.bailed;
    }
    if (args.score !== undefined && args.score !== null) {
      this.score = args.score;
    }
  }
};
JobInfo.prototype = {};
JobInfo.prototype.read = function(input) {
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
        this.job_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.I32) {
        this.user_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.I32) {
        this.search_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.I32) {
        this.status = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.I32) {
        this.priority = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 6:
      if (ftype == Thrift.Type.STRING) {
        this.callback = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 7:
      if (ftype == Thrift.Type.STRING) {
        this.time_added = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 8:
      if (ftype == Thrift.Type.I32) {
        this.agent_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 9:
      if (ftype == Thrift.Type.I32) {
        this.world_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 10:
      if (ftype == Thrift.Type.STRING) {
        this.agent_param = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 11:
      if (ftype == Thrift.Type.STRING) {
        this.world_param = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 12:
      if (ftype == Thrift.Type.STRING) {
        this.note = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 13:
      if (ftype == Thrift.Type.DOUBLE) {
        this.bail_threshold = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 14:
      if (ftype == Thrift.Type.STRING) {
        this.worker_name = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 15:
      if (ftype == Thrift.Type.I32) {
        this.instance_id = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 16:
      if (ftype == Thrift.Type.STRING) {
        this.worker_ip = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 17:
      if (ftype == Thrift.Type.STRING) {
        this.time_handed = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 18:
      if (ftype == Thrift.Type.I32) {
        this.seconds = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 19:
      if (ftype == Thrift.Type.I32) {
        this.steps = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 20:
      if (ftype == Thrift.Type.I32) {
        this.cycles = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 21:
      if (ftype == Thrift.Type.BOOL) {
        this.bailed = input.readBool().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 22:
      if (ftype == Thrift.Type.DOUBLE) {
        this.score = input.readDouble().value;
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

JobInfo.prototype.write = function(output) {
  output.writeStructBegin('JobInfo');
  if (this.job_id !== null && this.job_id !== undefined) {
    output.writeFieldBegin('job_id', Thrift.Type.I32, 1);
    output.writeI32(this.job_id);
    output.writeFieldEnd();
  }
  if (this.user_id !== null && this.user_id !== undefined) {
    output.writeFieldBegin('user_id', Thrift.Type.I32, 2);
    output.writeI32(this.user_id);
    output.writeFieldEnd();
  }
  if (this.search_id !== null && this.search_id !== undefined) {
    output.writeFieldBegin('search_id', Thrift.Type.I32, 3);
    output.writeI32(this.search_id);
    output.writeFieldEnd();
  }
  if (this.status !== null && this.status !== undefined) {
    output.writeFieldBegin('status', Thrift.Type.I32, 4);
    output.writeI32(this.status);
    output.writeFieldEnd();
  }
  if (this.priority !== null && this.priority !== undefined) {
    output.writeFieldBegin('priority', Thrift.Type.I32, 5);
    output.writeI32(this.priority);
    output.writeFieldEnd();
  }
  if (this.callback !== null && this.callback !== undefined) {
    output.writeFieldBegin('callback', Thrift.Type.STRING, 6);
    output.writeString(this.callback);
    output.writeFieldEnd();
  }
  if (this.time_added !== null && this.time_added !== undefined) {
    output.writeFieldBegin('time_added', Thrift.Type.STRING, 7);
    output.writeString(this.time_added);
    output.writeFieldEnd();
  }
  if (this.agent_id !== null && this.agent_id !== undefined) {
    output.writeFieldBegin('agent_id', Thrift.Type.I32, 8);
    output.writeI32(this.agent_id);
    output.writeFieldEnd();
  }
  if (this.world_id !== null && this.world_id !== undefined) {
    output.writeFieldBegin('world_id', Thrift.Type.I32, 9);
    output.writeI32(this.world_id);
    output.writeFieldEnd();
  }
  if (this.agent_param !== null && this.agent_param !== undefined) {
    output.writeFieldBegin('agent_param', Thrift.Type.STRING, 10);
    output.writeString(this.agent_param);
    output.writeFieldEnd();
  }
  if (this.world_param !== null && this.world_param !== undefined) {
    output.writeFieldBegin('world_param', Thrift.Type.STRING, 11);
    output.writeString(this.world_param);
    output.writeFieldEnd();
  }
  if (this.note !== null && this.note !== undefined) {
    output.writeFieldBegin('note', Thrift.Type.STRING, 12);
    output.writeString(this.note);
    output.writeFieldEnd();
  }
  if (this.bail_threshold !== null && this.bail_threshold !== undefined) {
    output.writeFieldBegin('bail_threshold', Thrift.Type.DOUBLE, 13);
    output.writeDouble(this.bail_threshold);
    output.writeFieldEnd();
  }
  if (this.worker_name !== null && this.worker_name !== undefined) {
    output.writeFieldBegin('worker_name', Thrift.Type.STRING, 14);
    output.writeString(this.worker_name);
    output.writeFieldEnd();
  }
  if (this.instance_id !== null && this.instance_id !== undefined) {
    output.writeFieldBegin('instance_id', Thrift.Type.I32, 15);
    output.writeI32(this.instance_id);
    output.writeFieldEnd();
  }
  if (this.worker_ip !== null && this.worker_ip !== undefined) {
    output.writeFieldBegin('worker_ip', Thrift.Type.STRING, 16);
    output.writeString(this.worker_ip);
    output.writeFieldEnd();
  }
  if (this.time_handed !== null && this.time_handed !== undefined) {
    output.writeFieldBegin('time_handed', Thrift.Type.STRING, 17);
    output.writeString(this.time_handed);
    output.writeFieldEnd();
  }
  if (this.seconds !== null && this.seconds !== undefined) {
    output.writeFieldBegin('seconds', Thrift.Type.I32, 18);
    output.writeI32(this.seconds);
    output.writeFieldEnd();
  }
  if (this.steps !== null && this.steps !== undefined) {
    output.writeFieldBegin('steps', Thrift.Type.I32, 19);
    output.writeI32(this.steps);
    output.writeFieldEnd();
  }
  if (this.cycles !== null && this.cycles !== undefined) {
    output.writeFieldBegin('cycles', Thrift.Type.I32, 20);
    output.writeI32(this.cycles);
    output.writeFieldEnd();
  }
  if (this.bailed !== null && this.bailed !== undefined) {
    output.writeFieldBegin('bailed', Thrift.Type.BOOL, 21);
    output.writeBool(this.bailed);
    output.writeFieldEnd();
  }
  if (this.score !== null && this.score !== undefined) {
    output.writeFieldBegin('score', Thrift.Type.DOUBLE, 22);
    output.writeDouble(this.score);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

BinInfo = function(args) {
  this.bin_id = null;
  this.name = null;
  this.version = null;
  this.package_hash = null;
  this.time_added = null;
  this.type = null;
  this.status = null;
  if (args) {
    if (args.bin_id !== undefined && args.bin_id !== null) {
      this.bin_id = args.bin_id;
    }
    if (args.name !== undefined && args.name !== null) {
      this.name = args.name;
    }
    if (args.version !== undefined && args.version !== null) {
      this.version = args.version;
    }
    if (args.package_hash !== undefined && args.package_hash !== null) {
      this.package_hash = args.package_hash;
    }
    if (args.time_added !== undefined && args.time_added !== null) {
      this.time_added = args.time_added;
    }
    if (args.type !== undefined && args.type !== null) {
      this.type = args.type;
    }
    if (args.status !== undefined && args.status !== null) {
      this.status = args.status;
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
        this.bin_id = input.readI32().value;
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
        this.package_hash = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.STRING) {
        this.time_added = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 6:
      if (ftype == Thrift.Type.I32) {
        this.type = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 7:
      if (ftype == Thrift.Type.I32) {
        this.status = input.readI32().value;
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
  if (this.bin_id !== null && this.bin_id !== undefined) {
    output.writeFieldBegin('bin_id', Thrift.Type.I32, 1);
    output.writeI32(this.bin_id);
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
  if (this.package_hash !== null && this.package_hash !== undefined) {
    output.writeFieldBegin('package_hash', Thrift.Type.STRING, 4);
    output.writeString(this.package_hash);
    output.writeFieldEnd();
  }
  if (this.time_added !== null && this.time_added !== undefined) {
    output.writeFieldBegin('time_added', Thrift.Type.STRING, 5);
    output.writeString(this.time_added);
    output.writeFieldEnd();
  }
  if (this.type !== null && this.type !== undefined) {
    output.writeFieldBegin('type', Thrift.Type.I32, 6);
    output.writeI32(this.type);
    output.writeFieldEnd();
  }
  if (this.status !== null && this.status !== undefined) {
    output.writeFieldBegin('status', Thrift.Type.I32, 7);
    output.writeI32(this.status);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

