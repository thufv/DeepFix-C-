using System;
using System.Collections;
using System.Windows.Forms;
using System.Drawing;
public class Program {
  ArrayList lstNumbers;
  private void QuestionAnswer_Load(object sender, EventArgs e) {
    txtQuestion.Enabled = false;
    txtQuestion.BackColor = Color.White;
    grpMultipleChoice.Enabled = false;
    grpSingleChoice.Enabled = false;
    btnCheckAnswer.Enabled = false;
    btnNext.Enabled = false;
    btnQuit.Enabled = false;
    if (txtStudentName.Text == "" && txtStudentNumber.Text == "" && txtModuleNumber.Text == "") {
      btnStart.Enabled = false;
      MessageBox.Show("You must enter your registration details at the upper right corner");
    }
    lstNumbers = new ArrayList();
    Random rndNumber = new Random();
    int count = 0;
    int maximumNumber = 69;
    btnCheckAnswer.Enabled = false;
    do {
      int number = (int)(rndNumber.NextDouble() * maximumNumber) + 1;
      lstNumbers.Add(number);
      if (!lstNumbers.Contains(number)) {
        lstNumbers.Add(number);
      }
      count++;
    } while ( count <= 15 * 70 );
    btnCheckAnswer_Click(sender, e);
    return lstNumbers;
  }
  void btnCheckAnswer_Click(object sender, EventArgs e) {
    /* inserted */
    int _23 = 23;
  }
  Button btnCheckAnswer,
  btnNext,
  btnQuit,
  btnStart,
  grpMultipleChoice,
  grpSingleChoice;
  TextBox txtStudentName,
  txtQuestion,
  txtStudentNumber,
  txtModuleNumber;
  static void Main() {}
}
