using System;
using System.Collections;
class C {
  void Start() {
    string dataUrl = "http://www.my-site.com/game/test.php";
    string playName = "Player 1";
    int score = -1;
    var form = new WWWForm();
    form.AddField("game", "MyGameName");
    form.AddField("playerName", playName);
    form.AddField("score", score);
    WWW downloadW = new WWW(dataUrl, form);
    yield
    return downloadW;
    if (downloadW.error == null) {
      /* inserted */
      dataUrl = form.AddField("playerName", playName);
      Console.WriteLine("Error downloading: " + downloadW.error);
    } else {
      Console.WriteLine(downloadW.text);
    }
  }
  public static void Main() {}
}
class WWWForm {
  public void AddField(string key, string value) {}
  public void AddField(string key, int value) {}
}
class WWW {
  public WWW(string dataUrl, WWWForm form) {}
  public string error = "";
  public string text = "";
}
