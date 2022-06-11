var gServerURL="http://localhost:8000"; // /JobCzar

function start()
{

}

function onSendSQLButton()
{
  sqlString=$('#sqlString').val();
  sendQuery(sqlString);
}

function myCallback(thingo) {
    console.log("CALLBACK")
    console.log(thingo)
}

function sendQuery(sqlString)
{
    var transport = new Thrift.Transport("http://test.com:8000/test")//, {useCORS: true});
    // var transport = new Thrift.TWebSocketTransport(gServerURL)//, {useCORS: true});
    var protocol  = new Thrift.TJSONProtocol(transport);
    var client    = new JobCzarClient(protocol);

    console.log("hello!");

    try {
        result = client.runSQL(sqlString);
        console.log("RESULT")
        console.log(result)
        $('#result').text(result);
        $('#result').css('color', 'black');
      } catch(error){
        console.log("ERROR")
        console.log(error)
        $('#result').text(error.why);
        $('#result').css('color', 'red');
      }

}