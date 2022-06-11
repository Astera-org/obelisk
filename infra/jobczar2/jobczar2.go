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
	"fmt"
	"github.com/rs/cors"
	"github.com/zenazn/goji"
	"net/http"
)

func sum(w http.ResponseWriter, r *http.Request) {
	println("I like kittens")
	fmt.Fprint(w, "I like turtles")
	w.Header().Set("Content-Type", "application/json")
	//w.Write("")
}

func main() {
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://test.com", "127.0.0.1", "localhost", "localhost:8000", "localhost:9009", "null"},
	})
	goji.Use(c.Handler)
	goji.Get("/test", sum)
	goji.Post("/test", sum)
	goji.Serve()
}
