
const gServerURLLocal = "http://localhost:8000/jobczar"; // localhost jobczar
const gServerURLProd = "http://192.168.1.41:9000/jobczar"; // production JobCzar

// toggle between localhost and prod
let local = true;

// thrift client that talks to the server
let gClient = null;

// map agent, world ids to names
let gBinInfos = null;

// TODO: some code organization:
// separate thrift client code
// how should we organize javascript? modules?

function getClient() {
    if (gClient == null) {
        console.log("getClient local,", local);
        const address = local ? gServerURLLocal : gServerURLProd;
        const transport = new Thrift.TXHRTransport(address, {useCORS: true});
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
        if (result instanceof Error) {
            errorAlert("runSql server error: " + result);
        } else {
            successAlert("Success! num results: " + result.length);
            generateResultsTable(result);
        }
    });
}

function addJob(agent_id, agent_param, world_id, world_param, note) {
    console.log("addJob agent:", agent_id, "agent config:", agent_param, "world:",
        world_id, "world config:", world_param, "note:", note);
    const client = getClient();
    client.addJob(agent_id, world_id, agent_param, world_param, -1, -1, note,function (result) {
        console.log("addJob result", result);
        if (result instanceof Error) {
            errorAlert("addJob server error: " + result);
        } else {
            successAlert("Success! New jod id: " + result);
        }
    });
}

function getBinName(bin_id) {
    if (!gBinInfos)
        return ""
    const binInfo = gBinInfos[bin_id];
    if (!binInfo)
        return "";
    return binInfo.name;
}

// convert a JobInfo object to a table row
function toHtml(ji) {
    return `
      <tr>
        <td>${ji.job_id}</td>
        <td>${getBinName(ji.agent_id)}/${ji.agent_id}</td>
        <td>${ji.agent_param}</td>
        <td>${getBinName(ji.world_id)}/${ji.world_id}</td>
        <td>${ji.world_param}</td>
        <td>${ji.score}</td>
        <td>${toStatus(ji.status)}</td>
        <td>${ji.search_id}</td>
        <td>${ji.time_added}</td>
        <td>${ji.note}</td>
        <td><button type="button" class="btn" id="${ji.job_id}">cancel</button></td>
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

// each row is a map from column name to string value
function generateResultsTable(rows) {
    if (!rows)
        return;

    const table = $('#results_table > tbody');
    table.empty();

    const head = $('#results_table > thead');
    head.empty();

    // generate header from column names
    const columns = Object.keys(rows[0]);
    head.append(`<tr>`);
    columns.forEach(function (col) {
        head.append(`<th scope="col">${col}</th>`);
    });
    head.append(`</tr>`);

    rows.forEach(function(row) {
        table.append(`<tr>`);
        columns.forEach(function (col) {
            table.append(`<td>${row[col]}</td>`);
        });
        table.append(`</tr>`);
    });
}

function generateJobsTable(jobInfos) {
    const table = $('#jobs_table > tbody');
    table.empty();
    // clear out old click listeners or they pile up
    table.off('click');

    const head = $('#jobs_table > thead');
    head.empty();
    head.append(`
        <tr>
            <th scope="col">Job id</th>
            <th scope="col">Agent</th>
            <th scope="col">Agent param</th>
            <th scope="col">World</th>
            <th scope="col">World param</th>
            <th scope="col">Score</th>
            <th scope="col">Status</th>
            <th scope="col">Search id</th>
            <th scope="col">Time added</th>
            <th scope="col">Note</th>
            <th scope="col">Cancel</th>
            <!-- Any other useful columns? -->
        </tr>
    `);

    jobInfos.forEach(function (ji) {
        table.append(toHtml(ji));
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
    const filter_by = "";

    client.queryJobs(filter_by, function (result) {
        if (result instanceof Error) {
            errorAlert("queryJobs server error: ", result);
        } else {
            console.log("queryJobs result", result);
            successAlert("Received jobs data from server");
            generateJobsTable(result);
        }
    });
}

function populateOptions(selectElem, binInfos) {
    selectElem.empty();
    binInfos.forEach(function (bi) {
        selectElem.append('<option value="' + bi.bin_id + '">' + bi.name + '</option>');
    });
}

function getBinInfos() {
    const client = getClient();
    // status=0 means current
    const filter_by = "status=0";

    client.getBinInfos(filter_by,function (binInfos) {
        if (binInfos instanceof Error) {
            errorAlert("getBinInfos server error: ", binInfos);
        } else {
            console.log("getBinInfos result", binInfos);
            // convert the list of infos to map from id to info
            gBinInfos = binInfos.reduce((map, obj) => (map[obj.bin_id] = obj, map), {});
            successAlert("Received bin info data from server");
            // type 0 means agent, type 1 means world
            const agents = binInfos.filter(bi => bi.type === 0);
            const worlds = binInfos.filter(bi => bi.type === 1);
            const agentSelect = $("#agent");
            const worldSelect = $("#world");
            populateOptions(agentSelect, agents);
            populateOptions(worldSelect, worlds);

            // we call this here because we need the bin info first
            queryJobs();
        }
    });
}

function setServerText() {
    const serverText = $("#serverText");
    serverText.text(local ? "Localhost" : "Production");
    // also set the hidden checkbox so it saves state
    const serverToggleCb = $("#serverToggleCheckbox");
    serverToggleCb.prop("checked", !local);
}

function successAlert(text) {
    const error = $(".alert-danger");
    error.hide();
    const success = $(".alert-success");
    success.show();
    success.text(text);
}

function errorAlert(text) {
    const success = $(".alert-success");
    success.hide();
    const error = $(".alert-danger");
    error.show();
    error.text(text);
    console.error(text);
}

function toggleServer() {
    local = !local;
    // reset the client to point to the new address
    gClient = null;
    setServerText();
    getBinInfos();
}

$(function() {
    console.log("document ready");

    // setup server toggle
    const serverToggleCb = $("#serverToggleCheckbox");
    const isChecked = serverToggleCb.is(':checked');
    // the browser saves the last state of the checkbox so set it based on it
    local = !isChecked;
    setServerText();

    getBinInfos();

    $("#serverToggleLocalhost").click(function (event) {
        if (local) {
            console.log("already local");
            return;
        }
        toggleServer();
    });
    $("#serverToggleProduction").click(function (event) {
        if (!local) {
            console.log("already prod");
            return;
        }
        toggleServer();
    });

    // setup add job form
    $("#add_job_form").submit(function(event) {
        event.preventDefault();
        const agentSelect = $("#agent");
        const agent_id = agentSelect.val();
        const agent_param = $("#agent-config").val();
        const worldSelect = $("#world");
        const world_id = worldSelect.val();
        const world_param = $("#world-config").val();
        const note = $("#note").val();

        addJob(agent_id, agent_param, world_id, world_param, note);
    });

    // setup raw sql form
    $("#submit_sql_form").submit(function(event) {
        event.preventDefault();
        const sqlString = $("#sqlString").val();
        runSql(sqlString);
    });

});
