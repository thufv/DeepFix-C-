class Program {
  static void Main(string[] args) {
    /* inserted */
    int _28 = 3;
    string destination = "default";
    switch (destination) {
    case "Inventory":
      AddToInventory();
      break;
    case "Hotbar":
      AddToHotbar();
      break;
    default:
      Debug.Log("Destination Unknown");
    }
  }
  static void AddToInventory() {}
  static void AddToHotbar() {}
}
class Debug {
  public static void Log(string v) {}
}
