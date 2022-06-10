// Code generated by Thrift Compiler (0.16.0). DO NOT EDIT.

package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	thrift "github.com/apache/thrift/lib/go/thrift"
	"infra"
)

var _ = infra.GoUnusedProtection__

func Usage() {
  fmt.Fprintln(os.Stderr, "Usage of ", os.Args[0], " [-h host:port] [-u url] [-f[ramed]] function [arg1 [arg2...]]:")
  flag.PrintDefaults()
  fmt.Fprintln(os.Stderr, "\nFunctions:")
  fmt.Fprintln(os.Stderr, "  Job fetchWork(string workerName, string instanceName)")
  fmt.Fprintln(os.Stderr, "  bool submitResult(ResultWork result)")
  fmt.Fprintln(os.Stderr, "  i32 addJob(string agentName, string worldName, string agentCfg, string worldCfg, i32 priority, i32 userID)")
  fmt.Fprintln(os.Stderr, "  string runSQL(string query)")
  fmt.Fprintln(os.Stderr, "  bool removeJob(i32 jobID)")
  fmt.Fprintln(os.Stderr)
  os.Exit(0)
}

type httpHeaders map[string]string

func (h httpHeaders) String() string {
  var m map[string]string = h
  return fmt.Sprintf("%s", m)
}

func (h httpHeaders) Set(value string) error {
  parts := strings.Split(value, ": ")
  if len(parts) != 2 {
    return fmt.Errorf("header should be of format 'Key: Value'")
  }
  h[parts[0]] = parts[1]
  return nil
}

func main() {
  flag.Usage = Usage
  var host string
  var port int
  var protocol string
  var urlString string
  var framed bool
  var useHttp bool
  headers := make(httpHeaders)
  var parsedUrl *url.URL
  var trans thrift.TTransport
  _ = strconv.Atoi
  _ = math.Abs
  flag.Usage = Usage
  flag.StringVar(&host, "h", "localhost", "Specify host and port")
  flag.IntVar(&port, "p", 9090, "Specify port")
  flag.StringVar(&protocol, "P", "binary", "Specify the protocol (binary, compact, simplejson, json)")
  flag.StringVar(&urlString, "u", "", "Specify the url")
  flag.BoolVar(&framed, "framed", false, "Use framed transport")
  flag.BoolVar(&useHttp, "http", false, "Use http")
  flag.Var(headers, "H", "Headers to set on the http(s) request (e.g. -H \"Key: Value\")")
  flag.Parse()
  
  if len(urlString) > 0 {
    var err error
    parsedUrl, err = url.Parse(urlString)
    if err != nil {
      fmt.Fprintln(os.Stderr, "Error parsing URL: ", err)
      flag.Usage()
    }
    host = parsedUrl.Host
    useHttp = len(parsedUrl.Scheme) <= 0 || parsedUrl.Scheme == "http" || parsedUrl.Scheme == "https"
  } else if useHttp {
    _, err := url.Parse(fmt.Sprint("http://", host, ":", port))
    if err != nil {
      fmt.Fprintln(os.Stderr, "Error parsing URL: ", err)
      flag.Usage()
    }
  }
  
  cmd := flag.Arg(0)
  var err error
  var cfg *thrift.TConfiguration = nil
  if useHttp {
    trans, err = thrift.NewTHttpClient(parsedUrl.String())
    if len(headers) > 0 {
      httptrans := trans.(*thrift.THttpClient)
      for key, value := range headers {
        httptrans.SetHeader(key, value)
      }
    }
  } else {
    portStr := fmt.Sprint(port)
    if strings.Contains(host, ":") {
           host, portStr, err = net.SplitHostPort(host)
           if err != nil {
                   fmt.Fprintln(os.Stderr, "error with host:", err)
                   os.Exit(1)
           }
    }
    trans = thrift.NewTSocketConf(net.JoinHostPort(host, portStr), cfg)
    if err != nil {
      fmt.Fprintln(os.Stderr, "error resolving address:", err)
      os.Exit(1)
    }
    if framed {
      trans = thrift.NewTFramedTransportConf(trans, cfg)
    }
  }
  if err != nil {
    fmt.Fprintln(os.Stderr, "Error creating transport", err)
    os.Exit(1)
  }
  defer trans.Close()
  var protocolFactory thrift.TProtocolFactory
  switch protocol {
  case "compact":
    protocolFactory = thrift.NewTCompactProtocolFactoryConf(cfg)
    break
  case "simplejson":
    protocolFactory = thrift.NewTSimpleJSONProtocolFactoryConf(cfg)
    break
  case "json":
    protocolFactory = thrift.NewTJSONProtocolFactory()
    break
  case "binary", "":
    protocolFactory = thrift.NewTBinaryProtocolFactoryConf(cfg)
    break
  default:
    fmt.Fprintln(os.Stderr, "Invalid protocol specified: ", protocol)
    Usage()
    os.Exit(1)
  }
  iprot := protocolFactory.GetProtocol(trans)
  oprot := protocolFactory.GetProtocol(trans)
  client := infra.NewJobCzarClient(thrift.NewTStandardClient(iprot, oprot))
  if err := trans.Open(); err != nil {
    fmt.Fprintln(os.Stderr, "Error opening socket to ", host, ":", port, " ", err)
    os.Exit(1)
  }
  
  switch cmd {
  case "fetchWork":
    if flag.NArg() - 1 != 2 {
      fmt.Fprintln(os.Stderr, "FetchWork requires 2 args")
      flag.Usage()
    }
    argvalue0 := flag.Arg(1)
    value0 := argvalue0
    argvalue1 := flag.Arg(2)
    value1 := argvalue1
    fmt.Print(client.FetchWork(context.Background(), value0, value1))
    fmt.Print("\n")
    break
  case "submitResult":
    if flag.NArg() - 1 != 1 {
      fmt.Fprintln(os.Stderr, "SubmitResult_ requires 1 args")
      flag.Usage()
    }
    arg20 := flag.Arg(1)
    mbTrans21 := thrift.NewTMemoryBufferLen(len(arg20))
    defer mbTrans21.Close()
    _, err22 := mbTrans21.WriteString(arg20)
    if err22 != nil {
      Usage()
      return
    }
    factory23 := thrift.NewTJSONProtocolFactory()
    jsProt24 := factory23.GetProtocol(mbTrans21)
    argvalue0 := infra.NewResultWork()
    err25 := argvalue0.Read(context.Background(), jsProt24)
    if err25 != nil {
      Usage()
      return
    }
    value0 := argvalue0
    fmt.Print(client.SubmitResult_(context.Background(), value0))
    fmt.Print("\n")
    break
  case "addJob":
    if flag.NArg() - 1 != 6 {
      fmt.Fprintln(os.Stderr, "AddJob requires 6 args")
      flag.Usage()
    }
    argvalue0 := flag.Arg(1)
    value0 := argvalue0
    argvalue1 := flag.Arg(2)
    value1 := argvalue1
    argvalue2 := flag.Arg(3)
    value2 := argvalue2
    argvalue3 := flag.Arg(4)
    value3 := argvalue3
    tmp4, err30 := (strconv.Atoi(flag.Arg(5)))
    if err30 != nil {
      Usage()
      return
    }
    argvalue4 := int32(tmp4)
    value4 := argvalue4
    tmp5, err31 := (strconv.Atoi(flag.Arg(6)))
    if err31 != nil {
      Usage()
      return
    }
    argvalue5 := int32(tmp5)
    value5 := argvalue5
    fmt.Print(client.AddJob(context.Background(), value0, value1, value2, value3, value4, value5))
    fmt.Print("\n")
    break
  case "runSQL":
    if flag.NArg() - 1 != 1 {
      fmt.Fprintln(os.Stderr, "RunSQL requires 1 args")
      flag.Usage()
    }
    argvalue0 := flag.Arg(1)
    value0 := argvalue0
    fmt.Print(client.RunSQL(context.Background(), value0))
    fmt.Print("\n")
    break
  case "removeJob":
    if flag.NArg() - 1 != 1 {
      fmt.Fprintln(os.Stderr, "RemoveJob requires 1 args")
      flag.Usage()
    }
    tmp0, err33 := (strconv.Atoi(flag.Arg(1)))
    if err33 != nil {
      Usage()
      return
    }
    argvalue0 := int32(tmp0)
    value0 := argvalue0
    fmt.Print(client.RemoveJob(context.Background(), value0))
    fmt.Print("\n")
    break
  case "":
    Usage()
    break
  default:
    fmt.Fprintln(os.Stderr, "Invalid function ", cmd)
  }
}
