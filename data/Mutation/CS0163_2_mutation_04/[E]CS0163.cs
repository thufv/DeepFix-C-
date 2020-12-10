class Program {
  static void Main(string[] args) {
    string destination = "_default"
    /* updated */
    ;
    switch (destination) {
    case "Inventory":
      AddToInventory();
      break;
    case "Hotbar":
      AddToHotbar();
    default:
      Debug.Log("Destination Unknown");
      break;
    }
  }
  static void AddToInventory() {}
  static void AddToHotbar() {}
}
class Debug {
  public static void Log(string v) {}
}
