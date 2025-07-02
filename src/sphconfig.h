#ifndef SPHCONFIG_H
#define SPHCONFIG_H


#include "vec3.h"

#include <QTreeWidget>


class SPH;


class SphConfig : public QTreeWidget
{

   Q_OBJECT

public:

   SphConfig(QWidget* parent = 0);


   void setSph(SPH *sph);


public slots:

   void readValuesFromSimulation();
   void writeValuesToSimulation();


protected:


   QTreeWidgetItem* mGravityX;
   QTreeWidgetItem* mGravityY;
   QTreeWidgetItem* mGravityZ;

   QTreeWidgetItem* mStiffness;
   QTreeWidgetItem* mViscosity;
   QTreeWidgetItem* mDamping;
   QTreeWidgetItem* mTimeStep;
   QTreeWidgetItem* mCflLimit;

   // Quiero añadir para modificar la densidad del fluido y G:
   QTreeWidgetItem* mGravConstant;
   QTreeWidgetItem* mRho0;

   SPH* mSph;
};

#endif // SPHCONFIG_H
