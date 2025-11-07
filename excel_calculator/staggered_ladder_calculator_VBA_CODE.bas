Attribute VB_Name = "Calculator"
Private Sub Worksheet_Change(ByVal Target As Range)
    Dim NumRungsCell As Range
    Set NumRungsCell = Range("B7")
    
    If Not Intersect(Target, NumRungsCell) Is Nothing Then
        Application.ScreenUpdating = False
        On Error Resume Next
        Range("A29:K49").AutoFilter Field:=11, Criteria1:="Yes"
        Range("A54:L74").AutoFilter Field:=12, Criteria1:="Yes"
        On Error GoTo 0
        Application.ScreenUpdating = True
    End If
End Sub

Private Sub Worksheet_Activate()
    On Error Resume Next
    Range("A29:K49").AutoFilter Field:=11, Criteria1:="Yes"
    Range("A54:L74").AutoFilter Field:=12, Criteria1:="Yes"
    On Error GoTo 0
End Sub