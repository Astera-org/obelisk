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
	"env"
)

var _ = env.GoUnusedProtection__

func Usage() {
  fmt.Fprintln(os.Stderr, "Usage of ", os.Args[0], " [-h host:port] [-u url] [-f[ramed]] function [arg1 [arg2...]]:")
  flag.PrintDefaults()
  fmt.Fprintln(os.Stderr, "\nFunctions:")
  fmt.Fprintln(os.Stderr, "   init(Space actionSpace, Space observationSpace)")
  fmt.Fprintln(os.Stderr, "  Actions step(Observations observations, string debug)")
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
  client := env.NewAgentClient(thrift.NewTStandardClient(iprot, oprot))
  if err := trans.Open(); err != nil {
    fmt.Fprintln(os.Stderr, "Error opening socket to ", host, ":", port, " ", err)
    os.Exit(1)
  }
  
  switch cmd {
  case "init":
    if flag.NArg() - 1 != 2 {
      fmt.Fprintln(os.Stderr, "Init requires 2 args")
      flag.Usage()
    }
    arg28 := flag.Arg(1)
    mbTrans29 := thrift.NewTMemoryBufferLen(len(arg28))
    defer mbTrans29.Close()
    _, err30 := mbTrans29.WriteString(arg28)
    if err30 != nil { 
      Usage()
      return
    }
    factory31 := thrift.NewTJSONProtocolFactory()
    jsProt32 := factory31.GetProtocol(mbTrans29)
    containerStruct0 := env.NewAgentInitArgs()
    err33 := containerStruct0.ReadField1(context.Background(), jsProt32)
    if err33 != nil {
      Usage()
      return
    }
    argvalue0 := containerStruct0.ActionSpace
    value0 := env.Space(argvalue0)
    arg34 := flag.Arg(2)
    mbTrans35 := thrift.NewTMemoryBufferLen(len(arg34))
    defer mbTrans35.Close()
    _, err36 := mbTrans35.WriteString(arg34)
    if err36 != nil { 
      Usage()
      return
    }
    factory37 := thrift.NewTJSONProtocolFactory()
    jsProt38 := factory37.GetProtocol(mbTrans35)
    containerStruct1 := env.NewAgentInitArgs()
    err39 := containerStruct1.ReadField2(context.Background(), jsProt38)
    if err39 != nil {
      Usage()
      return
    }
    argvalue1 := containerStruct1.ObservationSpace
    value1 := env.Space(argvalue1)
    fmt.Print(client.Init(context.Background(), value0, value1))
    fmt.Print("\n")
    break
  case "step":
    if flag.NArg() - 1 != 2 {
      fmt.Fprintln(os.Stderr, "Step requires 2 args")
      flag.Usage()
    }
    arg40 := flag.Arg(1)
    mbTrans41 := thrift.NewTMemoryBufferLen(len(arg40))
    defer mbTrans41.Close()
    _, err42 := mbTrans41.WriteString(arg40)
    if err42 != nil { 
      Usage()
      return
    }
    factory43 := thrift.NewTJSONProtocolFactory()
    jsProt44 := factory43.GetProtocol(mbTrans41)
    containerStruct0 := env.NewAgentStepArgs()
    err45 := containerStruct0.ReadField1(context.Background(), jsProt44)
    if err45 != nil {
      Usage()
      return
    }
    argvalue0 := containerStruct0.Observations
    value0 := env.Observations(argvalue0)
    argvalue1 := flag.Arg(2)
    value1 := argvalue1
    fmt.Print(client.Step(context.Background(), value0, value1))
    fmt.Print("\n")
    break
  case "":
    Usage()
    break
  default:
    fmt.Fprintln(os.Stderr, "Invalid function ", cmd)
  }
}
