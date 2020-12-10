public class D {
  public virtual void DoWork(int i) {
    int x = i
    /* ERROR: <missing ';'> */
  }
}
public abstract class E: D {
  public abstract override void DoWork(int i);
}
public class F: E {
  public override void DoWork(int i) {
    /* inserted */
    int _28 = 21;
  }
  public static void Main() {}
}
