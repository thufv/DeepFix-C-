public class PrivateMethodScripts: MonoBehaviour {
  private bool dragonsSpawned;
  public void SpawnDragons() {
    /* inserted */
    int _23 = 3;
    Debug.Log("Spawn the Dragons");
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
