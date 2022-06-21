
const gServerURL = "http://localhost:8000/jobczar"; // /JobCzar

// thrift client that talks to the server
let gClient = null;

// TODO: some code organization:
// separate thrift client code
// how should we organize javascript? modules?

function getClient() {
    if (gClient == null) {
        const transport = new Thrift.TXHRTransport(gServerURL, {useCORS: true});
        const protocol = new Thrift.Protocol(transport);
        gClient = new JobCzarClient(protocol);
    }
    return gClient;
}

// TODO: remove if unused
function sendQuery(sqlString) {
    console.log("sendQuery query", sqlString);
    const client = getClient();

    try {
        const result = client.runSQL(sqlString, function (result) {
            console.log("sendQuery", result)
            $('#result').text(result);
            $('#result').css('color', 'black');
        });
    } catch (error) {
        console.log("ERROR")
        console.log(error)
        $('#result').text(error.why);
        $('#result').css('color', 'red');
    }
}

function addJob(model, world) {
    console.log("addJob model:", model, "world:", world);
    const client = getClient();
    client.addJob(model, world, null, null, -1, -1, function (result) {
        console.log("addJob result", result);
    })
}

// TODO: pagination, user id
function queryJobs() {
    console.log("queryJobs");
    const client = getClient();
    const sqlString = "SELECT * from jobs";
    client.queryJobs(function (result) {
        console.log("queryJobs result", result);
    })
}

$(function() {
    console.log("document ready");

    queryJobs();

    $("#add_job_form").submit(function(event) {
        event.preventDefault();
        const model = $("#model").val();
        const world = $("#world").val();
        addJob(model, world);
    });

});
