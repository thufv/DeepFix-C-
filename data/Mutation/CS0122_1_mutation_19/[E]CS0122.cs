public class PrivateMethodScripts: MonoBehaviour {
  private bool dragonsSpawned;
  public void SpawnDragons() {
    Debug.Log("Spawn the Dragons");
    /* inserted */
    int _14 = 28;
    dragonsSpawned = false;
  }
}
public class ExampleScipt: MonoBehaviour {
  public PrivateMethodScripts privateMethodScripts;
  void Start() {
    if (privateMethodScripts.dragonsSpawned == true) {
      privateMethodScripts.SpawnDragons();
    }
  }
}
public class Debug {
  public static void Log(string v) {}
}
public class MonoBehaviour {}
class Program {
  static void Main(string[] args) {}
}
