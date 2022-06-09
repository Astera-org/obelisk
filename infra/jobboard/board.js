var gServerURL="http://localhost:9009";

function start()
{

}

function onSendSQLButton()
{
  sqlString=$('#sqlString').val();
  console.log("SEND SQL BUTTON")
  sendQuery(sqlString);
}

function myCallback(thingo) {
    console.log("CALLBACK")
    console.log(thingo)
}

function sendQuery(sqlString)
{
    var options = {}
    // options.useCORS = true
    var transport = new Thrift.Transport(gServerURL, options);
    var protocol  = new Thrift.TJSONProtocol(transport);
    console.log("part 1")
    var client    = new JobCzarClient(protocol);

    console.log("sending query yo")

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
    console.log("DONE")
}