var gServerURL="http://localhost";

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
    var transport = new Thrift.Transport(gServerURL);
    var protocol  = new Thrift.TJSONProtocol(transport);
    var client    = new JobCzarClient(protocol);

    try {
        result = client.runSQL(sqlString);
        $('#result').text(result);
        $('#result').css('color', 'black');
      } catch(error){
        $('#result').text(error.why);
        $('#result').css('color', 'red');
      }

}