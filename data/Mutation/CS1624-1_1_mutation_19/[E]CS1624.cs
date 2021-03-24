using System;
using System.Collections;
public class ExampleScript: MoniBehaviour {
  public GameObject timeMaster;
  void WaitCheck() {
    if (timeMaster.activeSelf) {
      yield
      return StartCorountine("Wait");
    }
  }
  private object StartCorountine(string v) {
    /* inserted */
    int _4 = 5;
    throw new NotImplementedException();
  }
}
public class GameObject {
  public bool activeSelf;
}
public class MoniBehaviour {}
public class Program {
  public static void Main() {}
}
