using System.Collections;
public class ChangeScene {
  public void ChangeToScene(string sceneToChangeTo) {
    float fadeTime = 1.0F;
    yield
    return new WaitForSeconds(fadeTime);
    Application.LoadLevel(sceneToChangeTo);
  }
  public static void Main() {}
}
class WaitForSeconds {
  public WaitForSeconds(float time) {
    /* inserted */
    int _26 = 9;
  }
}
class Application {
  public static void LoadLevel(string level) {}
}
