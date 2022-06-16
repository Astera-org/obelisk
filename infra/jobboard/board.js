var gServerURL="http://localhost:8000/jobczar"; // /JobCzar

function start()
{

}

function onSendSQLButton()
{
    sqlString=$('#sqlString').val();
    sendQuery(sqlString);
}

function sendQuery(sqlString)
{
    var transport = new Thrift.TXHRTransport(gServerURL, {useCORS: true});
    var protocol  = new Thrift.Protocol(transport);
    var client    = new JobCzarClient(protocol);

    console.log("sendQuery query", sqlString);

    try {
	result = client.runSQL(sqlString, function(result) {
	    console.log("async result", result)

	    $('#result').text(result);
            $('#result').css('color', 'black');
	});
      } catch(error){
        console.log("ERROR")
        console.log(error)
        $('#result').text(error.why);
        $('#result').css('color', 'red');
      }

}
