class Program {
  static void Main(string[] args) {
    string destination = "default";
    switch (destination) {
    case "Inventory":
      AddToInventory();
    case "Hotbar":
      AddToHotbar();
      break;
    default:
      Debug.Log("Destination Unknown");
      break;
    }
  }
  static void AddToInventory() {}
  static void AddToHotbar() {
    /* inserted */
    int _24 = 8;
  }
}
class Debug {
  public static void Log(string v) {}
}
