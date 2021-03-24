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
  static void AddToInventory() {
    /* inserted */
    int _9 = 22;
  }
  static void AddToHotbar() {}
}
class Debug {
  public static void Log(string v) {}
}
