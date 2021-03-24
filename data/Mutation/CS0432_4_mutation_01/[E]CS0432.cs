public class Automobile {
  public static int NumberOfWheels = 6
  /* updated */
  ;
  public static int SizeOfGasTank {
    get {
      return 15;
    }
  }
  public static void Drive() {}
  public static void Main() {}
}
public class cla {
  void func() {
    Automobile::Drive();
  }
}
