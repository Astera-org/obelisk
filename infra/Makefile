# Basic Go makefile

GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

DIRS=`go list ./...`

all: build

build: 
	$(GOBUILD) -v $(DIRS)

test: 
	$(GOTEST) -v $(DIRS)

clean: 
	$(GOCLEAN) ./...
