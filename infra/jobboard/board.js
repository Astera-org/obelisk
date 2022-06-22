
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
        queryJobs();
    })
}

// convert a json object to a table row
// see database.sql for the column names
// keep in sync with the table headers in jobboard.html
function toHtml(row) {
    return `
      <tr>
        <td>${row.job_id}</td>
        <td>${row.agent_name}</td>
        <td>${row.world_name}</td>
        <td>${row.score}</td>
        <td>${toStatus(row.status)}</td>
        <td><button type="button" class="btn" id="${row.job_id}">cancel</button></td>
        <!-- TODO: add more columns, cancel job, etc -->
       </tr>
     `;
}

function toStatus(status) {
    status = parseInt(status);
    // from database.sql: status (0-pending,1-working,2-complete,3-errored)
    switch (status) {
        case 0: return "pending"
        case 1: return "working"
        case 2: return "complete"
        case 3: return "errored"
    }
    return "unknown status " + status
}

function generateJobsTable(rows) {
    const table = $('#jobs_table > tbody');
    table.empty();
    // clear out old click listeners or they pile up
    table.off('click');

    rows.forEach(function (row) {
        table.append(toHtml(row));
    });

    // handle cancel job click
    table.on('click', 'button', function () {
        const job_id = parseInt($(this).attr('id'));
        cancelJob(job_id);
    });
}

function cancelJob(job_id) {
    console.log("cancel job", job_id);
    const client = getClient();
    client.removeJob(job_id, function (result) {
       console.log("remove job ", job_id, result);
       queryJobs();
    });
}

// TODO: pagination, fetch by user id, etc.
function queryJobs() {
    console.log("queryJobs");
    const client = getClient();

    client.queryJobs(function (result) {
        generateJobsTable(result);
    });
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
