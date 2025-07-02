/********************************************************************************
** Form generated from reading UI file 'widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QWidget>
#include "sphconfig.h"
#include "visualization.h"

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QGridLayout *gridLayout_3;
    QSplitter *splitter;
    QWidget *layoutWidget;
    QGridLayout *gridLayout_2;
    QGridLayout *gridLayout;
    QPlainTextEdit *mText;
    SphConfig *mAttributes;
    QPushButton *mQuit;
    QPushButton *mRun;
    QPushButton *mApply;
    QCheckBox *mDrawParticles;
    QCheckBox *mDrawVoxels;
    QSpacerItem *verticalSpacer;
    Visualization *mVisualization;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QString::fromUtf8("Widget"));
        Widget->resize(1171, 794);
        Widget->setStyleSheet(QString::fromUtf8("QWidget {\n"
"   background-color:rgb(33, 33, 33)\n"
"}\n"
"\n"
"QLabel {\n"
"   color: rgb(177, 177, 177); \n"
"}\n"
"\n"
"QListWidget {\n"
"   color: rgb(177, 177, 177); \n"
"   border: 1px solid rgb(63, 63, 63);\n"
"}\n"
"\n"
"QCheckBox {\n"
"   color: rgb(177, 177, 177); \n"
"   border: 1px solid rgb(63, 63, 63);\n"
"   text-align:left;\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"	background-color: rgb(80, 80, 80);\n"
"}\n"
"\n"
"QPushButton {\n"
"   color: rgb(177, 177, 177); \n"
"   border: 1px solid rgb(63, 63, 63);\n"
"}\n"
"\n"
"QMenu {\n"
"   color: rgb(177, 177, 177); \n"
"    background-color:rgb(33, 33, 33);\n"
"    border: 1px solid black;\n"
"}\n"
"\n"
"QMenu::item {\n"
"    background-color: transparent;\n"
"}\n"
"\n"
"QMenu::item:selected { \n"
"    background-color: rgb(63, 63, 63);\n"
"}\n"
"\n"
"QPlainTextEdit {\n"
"   color: rgb(177, 177, 177); \n"
"   border: 1px solid rgb(63, 63, 63);\n"
"}\n"
"\n"
"\n"
"\n"
"QTreeWidget {\n"
"   color: rgb(177, 177, 177); \n"
"   border: 1px solid rgb(63, "
                        "63, 63);\n"
"}\n"
"\n"
""));
        gridLayout_3 = new QGridLayout(Widget);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        splitter = new QSplitter(Widget);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        layoutWidget = new QWidget(splitter);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        gridLayout_2 = new QGridLayout(layoutWidget);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        mText = new QPlainTextEdit(layoutWidget);
        mText->setObjectName(QString::fromUtf8("mText"));

        gridLayout->addWidget(mText, 0, 0, 1, 3);

        mAttributes = new SphConfig(layoutWidget);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(1, QString::fromUtf8("value"));
        __qtreewidgetitem->setText(0, QString::fromUtf8("name"));
        mAttributes->setHeaderItem(__qtreewidgetitem);
        mAttributes->setObjectName(QString::fromUtf8("mAttributes"));
        mAttributes->setColumnCount(2);

        gridLayout->addWidget(mAttributes, 1, 0, 1, 3);

        mQuit = new QPushButton(layoutWidget);
        mQuit->setObjectName(QString::fromUtf8("mQuit"));

        gridLayout->addWidget(mQuit, 5, 2, 1, 1);

        mRun = new QPushButton(layoutWidget);
        mRun->setObjectName(QString::fromUtf8("mRun"));
        mRun->setStyleSheet(QString::fromUtf8("color: rgb(255, 255, 255); \n"
"border: 1px solid rgb(128, 128, 128);\n"
""));

        gridLayout->addWidget(mRun, 5, 1, 1, 1);

        mApply = new QPushButton(layoutWidget);
        mApply->setObjectName(QString::fromUtf8("mApply"));

        gridLayout->addWidget(mApply, 5, 0, 1, 1);

        mDrawParticles = new QCheckBox(layoutWidget);
        mDrawParticles->setObjectName(QString::fromUtf8("mDrawParticles"));
        mDrawParticles->setChecked(true);

        gridLayout->addWidget(mDrawParticles, 2, 0, 1, 3);

        mDrawVoxels = new QCheckBox(layoutWidget);
        mDrawVoxels->setObjectName(QString::fromUtf8("mDrawVoxels"));

        gridLayout->addWidget(mDrawVoxels, 4, 0, 1, 3);


        gridLayout_2->addLayout(gridLayout, 0, 0, 1, 1);

        verticalSpacer = new QSpacerItem(13, 100, QSizePolicy::Minimum, QSizePolicy::Preferred);

        gridLayout_2->addItem(verticalSpacer, 1, 0, 1, 1);

        splitter->addWidget(layoutWidget);
        mVisualization = new Visualization(splitter);
        mVisualization->setObjectName(QString::fromUtf8("mVisualization"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(mVisualization->sizePolicy().hasHeightForWidth());
        mVisualization->setSizePolicy(sizePolicy);
        mVisualization->setMinimumSize(QSize(800, 0));
        splitter->addWidget(mVisualization);

        gridLayout_3->addWidget(splitter, 0, 0, 1, 1);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "Smoothed particle hydrodynamics (SPH) | code by Matthias Varnholt (mueslee/haujobb) in 2016 | based on a webinar by Alan Heirich", nullptr));
        mQuit->setText(QCoreApplication::translate("Widget", "quit", nullptr));
        mRun->setText(QCoreApplication::translate("Widget", "run / pause", nullptr));
        mApply->setText(QCoreApplication::translate("Widget", "apply", nullptr));
        mDrawParticles->setText(QCoreApplication::translate("Widget", "draw particles", nullptr));
        mDrawVoxels->setText(QCoreApplication::translate("Widget", "draw voxels", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
