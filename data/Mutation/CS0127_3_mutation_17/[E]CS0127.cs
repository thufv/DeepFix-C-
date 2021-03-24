using System.Collections;
using System.Collections.Generic;
public class MethodeReturn {
  static void LongHypo(float a, float b) {
    float SommeCar = a * a + b * b;
    /* inserted */
    int _16 = b * b;
    return SommeCar;
  }
  void Start() {
    float result = LongHypo(3, 4);
    result++;
  }
  void Update() {}
  static void Main() {}
}
