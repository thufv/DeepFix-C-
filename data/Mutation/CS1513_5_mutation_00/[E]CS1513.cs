public class D {
  public virtual void DoWork(int i) {
    /* inserted */
    int _9 = 5;
  }
}
public abstract class E: D {
  public abstract override void DoWork(int i);
}
public class F: E {
  public override void DoWork(int i) {}
  public static void Main() {}
