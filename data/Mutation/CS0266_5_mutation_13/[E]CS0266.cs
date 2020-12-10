class Program {
  public static long sqlDefaultTime;
  public static void main() {
    int dd = 6000;
    SqlCommand sqlCmdDefaultTime = new SqlCommand("myQuery", sqlCon);
    sqlDefaultTime = sqlCmdDefaultTime.ExecuteReader();
    while (sqlDefaultTime > 0) {
      dd = sqlDefaultTime;
    }
  }
  private static SqlConnection sqlCon;
  public static void Main() {}
}
class SqlConnection {}
class SqlCommand {
  public SqlCommand(string cmd, SqlConnection conn) {
    /* inserted */
    int _7 = 22;
  }
  public long ExecuteReader() {
    return 10;
  }
}
