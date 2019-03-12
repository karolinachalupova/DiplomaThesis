%% --------------------------------------------------------------------
%% IES THESIS TEMPLATE
%% --------------------------------------------------------------------
%% version 2018-12-18 created by Tomas Havranek (IES) thanks to Jiri Roubal (CVUT) and Radek Fisher (IES), inspired by Suni Petal (Southampton). 

Formally, the template covers following FSS requirements:
>> number of characters per line
>> number of lines per page
>> font site
>> structure and appearance

In TeXnicCenter open project file (Thesis.tcp), so that the template opens up correctly. Otherwise compile Thesis.tex for personal info.

Adjust the jacket for your master thesis in 01_Frontmatter\jacket.tex.

Adjust the file Thesis.xmpdata as needed:
\Author{Firstname Lastname}
\Title{This is my title}
\Keywords{Keyword1, Keyword2, Keyword3, Keyword4}
\Subject{This is thesis subject}
\Publisher{Charles University}

For deeper changes in appearance rewrite macro file in Styles/Style.sty. 

Adjust the bibliography style in Styles/Stylebib.dbj or create a new one using software Makebst (or adjust the already made like newapa, elsevier-harv, etc)
For having separate bibliography list at the end of each chapter use package chapterbib (option sectionbib). 

Store graphics in folder Figures best in pdf/a-2u or jpeg.
