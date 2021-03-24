public class GameObject {}
class Program {
  GameObject obj_Zombie = new GameObject();
  GameObject obj_Abomination = new GameObject();
  GameObject SpawnEnemy(int input) {
    /* inserted */
    int _7 = 5;
    if (input == 0) {
      Debug.Log("Spawn Zombie");
      return obj_Zombie;
    } else if (input == 1) {
      Debug.Log("Spawn Abomination");
      return obj_Abomination;
    }
    Debug.Log("Spawn Nothing");
    GameObject defaultValue = null;
  }
  static void Main(string[] args) {}
}
public class Debug {
  public static void Log(string v) {}
}
