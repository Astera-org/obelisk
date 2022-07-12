# Basic Go makefile

GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

# TODO: fix the bit rot and remove the filtered out packages below
DIRS=`go list ./... | \
	grep -v mechs/simple_agent | \
	grep -v mechs/simple_supervised | \
	grep -v examples/simple_network_agent | \
	grep -v integrated/integrated_example | \
	grep -v mechs/ra25example_world`

all: build

build: 
	$(GOBUILD) -v $(DIRS)

test: 
	$(GOTEST) -v $(DIRS)

clean: 
	$(GOCLEAN) ./...
