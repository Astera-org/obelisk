var gServerURL="http://localhost:9009";

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
    var transport = new Thrift.Transport(gServerURL, {useCORS: true});
    var protocol  = new Thrift.TJSONProtocol(transport);
    var client    = new JobCzarClient(protocol);

    try {
        result = client.runSQL(sqlString, myCallback);
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