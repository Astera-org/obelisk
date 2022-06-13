// Copyright (c) 2016, Claudemiro Alves Feitosa Neto
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

package main

import (
	"context"
	"fmt"
	"github.com/rs/cors"
	"net/http"
	"net/http/httputil"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
	"github.com/zenazn/goji"
)

var defaultCtx = context.Background()

var gDatabase Database
var gConfig Config

// NewThriftHandlerFunc is a function that create a ready to use Apache Thrift Handler function
func NewThriftHandlerFunc(processor thrift.TProcessor,
	inPfactory, outPfactory thrift.TProtocolFactory) func(w http.ResponseWriter, r *http.Request) {

	return func(w http.ResponseWriter, r *http.Request) {
		fmt.Println("in thrift handler func", r.Body)
		reqDump, err := httputil.DumpRequest(r, true)
		if err != nil {
			fmt.Print(err)
		}
		fmt.Printf("REQUEST:\n%s\n", string(reqDump))

		transport := thrift.NewStreamTransport(r.Body, w)
		processor.Process(defaultCtx, inPfactory.GetProtocol(transport), outPfactory.GetProtocol(transport))
	}
}

func main() {
	gConfig.Load()
	gDatabase.Connect()

	// this is just a hack for localhost testing
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://test.com", "127.0.0.1", "localhost", "localhost:8000", "localhost:9009", "null"},
	})
	goji.Use(c.Handler)

	handler := RequestHandler{}
	processor := infra.NewJobCzarProcessor(handler)
	factory := thrift.NewTJSONProtocolFactory()

	goji.Post("/jobczar", NewThriftHandlerFunc(processor, factory, factory))

	goji.Serve()
}
