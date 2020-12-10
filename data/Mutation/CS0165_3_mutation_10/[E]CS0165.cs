using System;
public struct Vector2 {
  public event EventHandler trigger;
  public float X;
  public float Y;
  public Vector2 func() {
    Vector2 vector;
    vector.X = 1;
    vector.Y = 2;
    /* inserted */
    int _21 = 18;
    return vector;
  }
  public static void Main() {}
}
