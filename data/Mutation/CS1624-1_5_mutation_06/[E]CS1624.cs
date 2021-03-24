using System.Collections;
public class ChangeScene {
  public void ChangeToScene(string sceneToChangeTo) {
    float fadeTime = 1.0F;
    yield
    return new WaitForSeconds(fadeTime);
    /* inserted */
    int _4 = 6;
    Application.LoadLevel(sceneToChangeTo);
  }
  public static void Main() {}
}
class WaitForSeconds {
  public WaitForSeconds(float time) {}
}
class Application {
  public static void LoadLevel(string level) {}
}
