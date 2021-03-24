using System;
public struct Vector2 {
  public event EventHandler trigger;
  public float X;
  public float Y;
  public Vector2 func() {
    Vector2 vector;
    vector.X = 3
    /* updated */
    ;
    vector.Y = 2;
    return vector;
  }
  public static void Main() {}
}
