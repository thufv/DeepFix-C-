using System.Collections;
public class ChangeScene {
  public void ChangeToScene(string sceneToChangeTo) {
    float fadeTime = 1.0F;
    yield
    return new WaitForSeconds(fadeTime);
    Application.LoadLevel(sceneToChangeTo);
  }
  public static void Main() {
    /* inserted */
    int _16 = 30;
  }
}
class WaitForSeconds {
  public WaitForSeconds(float time) {}
}
class Application {
  public static void LoadLevel(string level) {}
}
