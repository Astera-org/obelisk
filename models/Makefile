# Basic Go makefile

GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

# TODO: fix the bit rot and remove the filtered out packages below
DIRS=`go list ./... | \
	  grep -v integrated/integrated_example`

all: build

build: 
	$(GOBUILD) -v $(DIRS)

test: 
	$(GOTEST) -v $(DIRS)

clean: 
	$(GOCLEAN) ./...
