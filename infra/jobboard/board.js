var gServerURL="http://localhost";

function sendQuery()
{
    var transport = new Thrift.Transport(gServerURL);
    var protocol  = new Thrift.TJSONProtocol(transport);
    var client    = new JobCzarClient(protocol);

    try {
        result = client.runSQL(sqlString);
        $('#result').text(result);
        $('#result').css('color', 'black');
      } catch(ouch){
        $('#result').text(ouch.why);
        $('#result').css('color', 'red');
      }

}