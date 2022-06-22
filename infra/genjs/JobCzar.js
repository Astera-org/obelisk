//
// Autogenerated by Thrift Compiler (0.16.0)
//
// DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
//
if (typeof Int64 === 'undefined' && typeof require === 'function') {
  var Int64 = require('node-int64');
}


//HELPER FUNCTIONS AND STRUCTURES

JobCzar_fetchWork_args = function(args) {
  this.workerName = null;
  this.instanceName = null;
  if (args) {
    if (args.workerName !== undefined && args.workerName !== null) {
      this.workerName = args.workerName;
    }
    if (args.instanceName !== undefined && args.instanceName !== null) {
      this.instanceName = args.instanceName;
    }
  }
};
JobCzar_fetchWork_args.prototype = {};
JobCzar_fetchWork_args.prototype.read = function(input) {
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
      if (ftype == Thrift.Type.STRING) {
        this.workerName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
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

JobCzar_fetchWork_args.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_fetchWork_args');
  if (this.workerName !== null && this.workerName !== undefined) {
    output.writeFieldBegin('workerName', Thrift.Type.STRING, 1);
    output.writeString(this.workerName);
    output.writeFieldEnd();
  }
  if (this.instanceName !== null && this.instanceName !== undefined) {
    output.writeFieldBegin('instanceName', Thrift.Type.STRING, 2);
    output.writeString(this.instanceName);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_fetchWork_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = new Job(args.success);
    }
  }
};
JobCzar_fetchWork_result.prototype = {};
JobCzar_fetchWork_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.STRUCT) {
        this.success = new Job();
        this.success.read(input);
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_fetchWork_result.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_fetchWork_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.STRUCT, 0);
    this.success.write(output);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_submitResult_args = function(args) {
  this.result = null;
  if (args) {
    if (args.result !== undefined && args.result !== null) {
      this.result = new ResultWork(args.result);
    }
  }
};
JobCzar_submitResult_args.prototype = {};
JobCzar_submitResult_args.prototype.read = function(input) {
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
      if (ftype == Thrift.Type.STRUCT) {
        this.result = new ResultWork();
        this.result.read(input);
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_submitResult_args.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_submitResult_args');
  if (this.result !== null && this.result !== undefined) {
    output.writeFieldBegin('result', Thrift.Type.STRUCT, 1);
    this.result.write(output);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_submitResult_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = args.success;
    }
  }
};
JobCzar_submitResult_result.prototype = {};
JobCzar_submitResult_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.BOOL) {
        this.success = input.readBool().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_submitResult_result.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_submitResult_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.BOOL, 0);
    output.writeBool(this.success);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_addJob_args = function(args) {
  this.agentName = null;
  this.worldName = null;
  this.agentCfg = null;
  this.worldCfg = null;
  this.priority = null;
  this.userID = null;
  if (args) {
    if (args.agentName !== undefined && args.agentName !== null) {
      this.agentName = args.agentName;
    }
    if (args.worldName !== undefined && args.worldName !== null) {
      this.worldName = args.worldName;
    }
    if (args.agentCfg !== undefined && args.agentCfg !== null) {
      this.agentCfg = args.agentCfg;
    }
    if (args.worldCfg !== undefined && args.worldCfg !== null) {
      this.worldCfg = args.worldCfg;
    }
    if (args.priority !== undefined && args.priority !== null) {
      this.priority = args.priority;
    }
    if (args.userID !== undefined && args.userID !== null) {
      this.userID = args.userID;
    }
  }
};
JobCzar_addJob_args.prototype = {};
JobCzar_addJob_args.prototype.read = function(input) {
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
      if (ftype == Thrift.Type.STRING) {
        this.agentName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.STRING) {
        this.worldName = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.STRING) {
        this.agentCfg = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.STRING) {
        this.worldCfg = input.readString().value;
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
      if (ftype == Thrift.Type.I32) {
        this.userID = input.readI32().value;
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

JobCzar_addJob_args.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_addJob_args');
  if (this.agentName !== null && this.agentName !== undefined) {
    output.writeFieldBegin('agentName', Thrift.Type.STRING, 1);
    output.writeString(this.agentName);
    output.writeFieldEnd();
  }
  if (this.worldName !== null && this.worldName !== undefined) {
    output.writeFieldBegin('worldName', Thrift.Type.STRING, 2);
    output.writeString(this.worldName);
    output.writeFieldEnd();
  }
  if (this.agentCfg !== null && this.agentCfg !== undefined) {
    output.writeFieldBegin('agentCfg', Thrift.Type.STRING, 3);
    output.writeString(this.agentCfg);
    output.writeFieldEnd();
  }
  if (this.worldCfg !== null && this.worldCfg !== undefined) {
    output.writeFieldBegin('worldCfg', Thrift.Type.STRING, 4);
    output.writeString(this.worldCfg);
    output.writeFieldEnd();
  }
  if (this.priority !== null && this.priority !== undefined) {
    output.writeFieldBegin('priority', Thrift.Type.I32, 5);
    output.writeI32(this.priority);
    output.writeFieldEnd();
  }
  if (this.userID !== null && this.userID !== undefined) {
    output.writeFieldBegin('userID', Thrift.Type.I32, 6);
    output.writeI32(this.userID);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_addJob_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = args.success;
    }
  }
};
JobCzar_addJob_result.prototype = {};
JobCzar_addJob_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.I32) {
        this.success = input.readI32().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_addJob_result.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_addJob_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.I32, 0);
    output.writeI32(this.success);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_runSQL_args = function(args) {
  this.query = null;
  if (args) {
    if (args.query !== undefined && args.query !== null) {
      this.query = args.query;
    }
  }
};
JobCzar_runSQL_args.prototype = {};
JobCzar_runSQL_args.prototype.read = function(input) {
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
      if (ftype == Thrift.Type.STRING) {
        this.query = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_runSQL_args.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_runSQL_args');
  if (this.query !== null && this.query !== undefined) {
    output.writeFieldBegin('query', Thrift.Type.STRING, 1);
    output.writeString(this.query);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_runSQL_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = args.success;
    }
  }
};
JobCzar_runSQL_result.prototype = {};
JobCzar_runSQL_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.STRING) {
        this.success = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_runSQL_result.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_runSQL_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.STRING, 0);
    output.writeString(this.success);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_removeJob_args = function(args) {
  this.jobID = null;
  if (args) {
    if (args.jobID !== undefined && args.jobID !== null) {
      this.jobID = args.jobID;
    }
  }
};
JobCzar_removeJob_args.prototype = {};
JobCzar_removeJob_args.prototype.read = function(input) {
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
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_removeJob_args.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_removeJob_args');
  if (this.jobID !== null && this.jobID !== undefined) {
    output.writeFieldBegin('jobID', Thrift.Type.I32, 1);
    output.writeI32(this.jobID);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_removeJob_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = args.success;
    }
  }
};
JobCzar_removeJob_result.prototype = {};
JobCzar_removeJob_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.BOOL) {
        this.success = input.readBool().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_removeJob_result.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_removeJob_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.BOOL, 0);
    output.writeBool(this.success);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_queryJobs_args = function(args) {
};
JobCzar_queryJobs_args.prototype = {};
JobCzar_queryJobs_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    input.skip(ftype);
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_queryJobs_args.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_queryJobs_args');
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzar_queryJobs_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = Thrift.copyList(args.success, [Thrift.copyMap, null]);
    }
  }
};
JobCzar_queryJobs_result.prototype = {};
JobCzar_queryJobs_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.LIST) {
        this.success = [];
        var _rtmp31 = input.readListBegin();
        var _size0 = _rtmp31.size || 0;
        for (var _i2 = 0; _i2 < _size0; ++_i2) {
          var elem3 = null;
          elem3 = {};
          var _rtmp35 = input.readMapBegin();
          var _size4 = _rtmp35.size || 0;
          for (var _i6 = 0; _i6 < _size4; ++_i6) {
            if (_i6 > 0 ) {
              if (input.rstack.length > input.rpos[input.rpos.length -1] + 1) {
                input.rstack.pop();
              }
            }
            var key7 = null;
            var val8 = null;
            key7 = input.readString().value;
            val8 = input.readString().value;
            elem3[key7] = val8;
          }
          input.readMapEnd();
          this.success.push(elem3);
        }
        input.readListEnd();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

JobCzar_queryJobs_result.prototype.write = function(output) {
  output.writeStructBegin('JobCzar_queryJobs_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.LIST, 0);
    output.writeListBegin(Thrift.Type.MAP, this.success.length);
    for (var iter9 in this.success) {
      if (this.success.hasOwnProperty(iter9)) {
        iter9 = this.success[iter9];
        output.writeMapBegin(Thrift.Type.STRING, Thrift.Type.STRING, Thrift.objectLength(iter9));
        for (var kiter10 in iter9) {
          if (iter9.hasOwnProperty(kiter10)) {
            var viter11 = iter9[kiter10];
            output.writeString(kiter10);
            output.writeString(viter11);
          }
        }
        output.writeMapEnd();
      }
    }
    output.writeListEnd();
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

JobCzarClient = function(input, output) {
  this.input = input;
  this.output = (!output) ? input : output;
  this.seqid = 0;
};
JobCzarClient.prototype = {};

JobCzarClient.prototype.fetchWork = function(workerName, instanceName, callback) {
  this.send_fetchWork(workerName, instanceName, callback); 
  if (!callback) {
    return this.recv_fetchWork();
  }
};

JobCzarClient.prototype.send_fetchWork = function(workerName, instanceName, callback) {
  var params = {
    workerName: workerName,
    instanceName: instanceName
  };
  var args = new JobCzar_fetchWork_args(params);
  try {
    this.output.writeMessageBegin('fetchWork', Thrift.MessageType.CALL, this.seqid);
    args.write(this.output);
    this.output.writeMessageEnd();
    if (callback) {
      var self = this;
      this.output.getTransport().flush(true, function() {
        var result = null;
        try {
          result = self.recv_fetchWork();
        } catch (e) {
          result = e;
        }
        callback(result);
      });
    } else {
      return this.output.getTransport().flush();
    }
  }
  catch (e) {
    if (typeof this.output.getTransport().reset === 'function') {
      this.output.getTransport().reset();
    }
    throw e;
  }
};

JobCzarClient.prototype.recv_fetchWork = function() {
  var ret = this.input.readMessageBegin();
  var mtype = ret.mtype;
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(this.input);
    this.input.readMessageEnd();
    throw x;
  }
  var result = new JobCzar_fetchWork_result();
  result.read(this.input);
  this.input.readMessageEnd();

  if (null !== result.success) {
    return result.success;
  }
  throw 'fetchWork failed: unknown result';
};

JobCzarClient.prototype.submitResult = function(result, callback) {
  this.send_submitResult(result, callback); 
  if (!callback) {
    return this.recv_submitResult();
  }
};

JobCzarClient.prototype.send_submitResult = function(result, callback) {
  var params = {
    result: result
  };
  var args = new JobCzar_submitResult_args(params);
  try {
    this.output.writeMessageBegin('submitResult', Thrift.MessageType.CALL, this.seqid);
    args.write(this.output);
    this.output.writeMessageEnd();
    if (callback) {
      var self = this;
      this.output.getTransport().flush(true, function() {
        var result = null;
        try {
          result = self.recv_submitResult();
        } catch (e) {
          result = e;
        }
        callback(result);
      });
    } else {
      return this.output.getTransport().flush();
    }
  }
  catch (e) {
    if (typeof this.output.getTransport().reset === 'function') {
      this.output.getTransport().reset();
    }
    throw e;
  }
};

JobCzarClient.prototype.recv_submitResult = function() {
  var ret = this.input.readMessageBegin();
  var mtype = ret.mtype;
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(this.input);
    this.input.readMessageEnd();
    throw x;
  }
  var result = new JobCzar_submitResult_result();
  result.read(this.input);
  this.input.readMessageEnd();

  if (null !== result.success) {
    return result.success;
  }
  throw 'submitResult failed: unknown result';
};

JobCzarClient.prototype.addJob = function(agentName, worldName, agentCfg, worldCfg, priority, userID, callback) {
  this.send_addJob(agentName, worldName, agentCfg, worldCfg, priority, userID, callback); 
  if (!callback) {
    return this.recv_addJob();
  }
};

JobCzarClient.prototype.send_addJob = function(agentName, worldName, agentCfg, worldCfg, priority, userID, callback) {
  var params = {
    agentName: agentName,
    worldName: worldName,
    agentCfg: agentCfg,
    worldCfg: worldCfg,
    priority: priority,
    userID: userID
  };
  var args = new JobCzar_addJob_args(params);
  try {
    this.output.writeMessageBegin('addJob', Thrift.MessageType.CALL, this.seqid);
    args.write(this.output);
    this.output.writeMessageEnd();
    if (callback) {
      var self = this;
      this.output.getTransport().flush(true, function() {
        var result = null;
        try {
          result = self.recv_addJob();
        } catch (e) {
          result = e;
        }
        callback(result);
      });
    } else {
      return this.output.getTransport().flush();
    }
  }
  catch (e) {
    if (typeof this.output.getTransport().reset === 'function') {
      this.output.getTransport().reset();
    }
    throw e;
  }
};

JobCzarClient.prototype.recv_addJob = function() {
  var ret = this.input.readMessageBegin();
  var mtype = ret.mtype;
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(this.input);
    this.input.readMessageEnd();
    throw x;
  }
  var result = new JobCzar_addJob_result();
  result.read(this.input);
  this.input.readMessageEnd();

  if (null !== result.success) {
    return result.success;
  }
  throw 'addJob failed: unknown result';
};

JobCzarClient.prototype.runSQL = function(query, callback) {
  this.send_runSQL(query, callback); 
  if (!callback) {
    return this.recv_runSQL();
  }
};

JobCzarClient.prototype.send_runSQL = function(query, callback) {
  var params = {
    query: query
  };
  var args = new JobCzar_runSQL_args(params);
  try {
    this.output.writeMessageBegin('runSQL', Thrift.MessageType.CALL, this.seqid);
    args.write(this.output);
    this.output.writeMessageEnd();
    if (callback) {
      var self = this;
      this.output.getTransport().flush(true, function() {
        var result = null;
        try {
          result = self.recv_runSQL();
        } catch (e) {
          result = e;
        }
        callback(result);
      });
    } else {
      return this.output.getTransport().flush();
    }
  }
  catch (e) {
    if (typeof this.output.getTransport().reset === 'function') {
      this.output.getTransport().reset();
    }
    throw e;
  }
};

JobCzarClient.prototype.recv_runSQL = function() {
  var ret = this.input.readMessageBegin();
  var mtype = ret.mtype;
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(this.input);
    this.input.readMessageEnd();
    throw x;
  }
  var result = new JobCzar_runSQL_result();
  result.read(this.input);
  this.input.readMessageEnd();

  if (null !== result.success) {
    return result.success;
  }
  throw 'runSQL failed: unknown result';
};

JobCzarClient.prototype.removeJob = function(jobID, callback) {
  this.send_removeJob(jobID, callback); 
  if (!callback) {
    return this.recv_removeJob();
  }
};

JobCzarClient.prototype.send_removeJob = function(jobID, callback) {
  var params = {
    jobID: jobID
  };
  var args = new JobCzar_removeJob_args(params);
  try {
    this.output.writeMessageBegin('removeJob', Thrift.MessageType.CALL, this.seqid);
    args.write(this.output);
    this.output.writeMessageEnd();
    if (callback) {
      var self = this;
      this.output.getTransport().flush(true, function() {
        var result = null;
        try {
          result = self.recv_removeJob();
        } catch (e) {
          result = e;
        }
        callback(result);
      });
    } else {
      return this.output.getTransport().flush();
    }
  }
  catch (e) {
    if (typeof this.output.getTransport().reset === 'function') {
      this.output.getTransport().reset();
    }
    throw e;
  }
};

JobCzarClient.prototype.recv_removeJob = function() {
  var ret = this.input.readMessageBegin();
  var mtype = ret.mtype;
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(this.input);
    this.input.readMessageEnd();
    throw x;
  }
  var result = new JobCzar_removeJob_result();
  result.read(this.input);
  this.input.readMessageEnd();

  if (null !== result.success) {
    return result.success;
  }
  throw 'removeJob failed: unknown result';
};

JobCzarClient.prototype.queryJobs = function(callback) {
  this.send_queryJobs(callback); 
  if (!callback) {
    return this.recv_queryJobs();
  }
};

JobCzarClient.prototype.send_queryJobs = function(callback) {
  var args = new JobCzar_queryJobs_args();
  try {
    this.output.writeMessageBegin('queryJobs', Thrift.MessageType.CALL, this.seqid);
    args.write(this.output);
    this.output.writeMessageEnd();
    if (callback) {
      var self = this;
      this.output.getTransport().flush(true, function() {
        var result = null;
        try {
          result = self.recv_queryJobs();
        } catch (e) {
          result = e;
        }
        callback(result);
      });
    } else {
      return this.output.getTransport().flush();
    }
  }
  catch (e) {
    if (typeof this.output.getTransport().reset === 'function') {
      this.output.getTransport().reset();
    }
    throw e;
  }
};

JobCzarClient.prototype.recv_queryJobs = function() {
  var ret = this.input.readMessageBegin();
  var mtype = ret.mtype;
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(this.input);
    this.input.readMessageEnd();
    throw x;
  }
  var result = new JobCzar_queryJobs_result();
  result.read(this.input);
  this.input.readMessageEnd();

  if (null !== result.success) {
    return result.success;
  }
  throw 'queryJobs failed: unknown result';
};
