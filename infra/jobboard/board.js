
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

function runSql(sqlString) {
    console.log("submitSql:", sqlString);
    const client = getClient();
    client.runSQL(sqlString, function (result) {
        console.log("runSQL result", result);
        $('#result').text(result);
    });
}

function addJob(agentID, agentCfg, worldID, worldCfg, note) {
    console.log("addJob agent:", agentID, "agent config:", agentCfg, "world:",
        worldID, "world config:", worldCfg, "note:", note);
    const client = getClient();
    client.addJob(agentID, worldID, agentCfg, worldCfg, -1, -1, note,function (result) {
        console.log("addJob result", result);
        queryJobs();
    });
}

// convert a json object to a table row
// see database.sql for the column names
// keep in sync with the table headers in jobboard.html
function toHtml(row) {
    return `
      <tr>
        <td>${row.job_id}</td>
        <td>${row.agent_id}</td>
        <td>${row.world_id}</td>
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
    const client = getClient();

    client.queryJobs(function (result) {
        console.log("queryJobs result", result);
        generateJobsTable(result);
    });
}

function populateOptions(selectElem, binInfos) {
    binInfos.forEach(function (bi) {
        selectElem.append('<option value="' + bi.binID + '">' + bi.name + '</option>');
    });
}

function getBinInfos() {
    const client = getClient();

    client.getBinInfos(function (binInfos) {
        console.log("getBinInfos result", binInfos);
        // type 0 means agent, type 1 means world
        const agents = binInfos.filter(bi => bi.type === 0);
        const worlds = binInfos.filter(bi => bi.type === 1);
        const agentSelect = $("#agent");
        const worldSelect = $("#world");
        populateOptions(agentSelect, agents);
        populateOptions(worldSelect, worlds);
    });
}

$(function() {
    console.log("document ready");

    queryJobs();

    getBinInfos();

    // setup add job form
    $("#add_job_form").submit(function(event) {
        event.preventDefault();
        const agentSelect = $("#agent");
        const agentID = agentSelect.val();
        const agentCfg = $("#agent-config").val();
        const worldSelect = $("#world");
        const worldID = worldSelect.val();
        const worldCfg = $("#world-config").val();
        const note = $("#note").val();

        addJob(agentID, agentCfg, worldID, worldCfg, note);
    });

    // setup raw sql form
    $("#submit_sql_form").submit(function(event) {
        event.preventDefault();
        const sqlString = $("#sqlString").val();
        runSql(sqlString);
    });

});
