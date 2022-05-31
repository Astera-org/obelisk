#include <iostream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "thrift_agent_client.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace env;

AgentClient* setup_client() {
    std::shared_ptr<TTransport> socket(new TSocket("localhost", 9090));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    AgentClient* client = new AgentClient(protocol);
    transport->open();
    return client;
}

int test() {

    AgentClient* agent = setup_client();

    Shape shape;
    shape.shape = {1};
    SpaceSpec actionSpec;
    actionSpec.shape = shape;
    Space actionSpace = {
	{"move", actionSpec}
    };

    SpaceSpec observationSpec;
    observationSpec.shape = shape;
    Space observationSpace = {
	{"move", observationSpec}
    };

    map<string, string> ret;
    agent->init(ret, actionSpace, observationSpace);
    cout << "init returned: " << endl;
    for (auto const& p : ret) {
	cout << p.first << ": " << p.second << endl;
    }

    ETensor et;
    et.shape = shape;
    et.values = {1, 2, 3};

    Observations observations = {
	{"world", et}
    };

    Actions actions;

    agent->step(actions, observations, "debug_string");

    for (auto const& p: actions) {
	cout << "action is: " << p.first << ": " << p.second << endl;
    }

    return 0;
}

/*
int main() {
    return test();
}
*/
