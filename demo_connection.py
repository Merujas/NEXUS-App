"""
NEXUS - Demo de Conexión entre Módulos
========================================

Este script demuestra cómo database_setup.py y economy_engine.py se conectan
para crear el sistema completo de incentivos conductuales.
"""

from database_setup import init_database, User, WithdrawalMethod
from economy_engine import NexusBank, compare_withdrawal_methods
from sqlalchemy.orm import Session


def demo_complete_flow():
    """
    Demuestra el flujo completo:
    1. Crear base de datos
    2. Crear usuario de prueba
    3. Otorgar puntos (recompensa publicitaria)
    4. Comparar métodos de retiro (diseño conductual)
    5. Procesar retiro a Kraken (incentivo)
    """
    print("NEXUS - Demo de Conexion de Modulos")
    print("=" * 60)
    
    # 1. Inicializar base de datos
    print("\n[Paso 1] Inicializando base de datos...")
    engine, SessionLocal = init_database("nexus_demo.sqlite")
    db: Session = SessionLocal()
    print("[OK] Base de datos creada: nexus_demo.sqlite")
    
    try:
        # 2. Crear o obtener usuario de prueba
        print("\n[Paso 2] Creando/obteniendo usuario de prueba...")
        test_user = db.query(User).filter(User.email == "demo@nexus.com").first()
        if not test_user:
            test_user = User(
                email="demo@nexus.com",
                nexus_points=0.0,
                fiat_balance=0.0,
                reputation_level=1
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
            print(f"[OK] Usuario creado: {test_user.email} (ID: {test_user.id})")
        else:
            # Resetear saldos para demo limpio
            test_user.nexus_points = 0.0
            test_user.fiat_balance = 0.0
            db.commit()
            db.refresh(test_user)
            print(f"[OK] Usuario existente encontrado: {test_user.email} (ID: {test_user.id})")
        
        # 3. Inicializar NexusBank
        print("\n[Paso 3] Inicializando NexusBank...")
        bank = NexusBank(db)
        print("[OK] NexusBank listo")
        
        # 4. Otorgar puntos por recompensa publicitaria
        print("\n[Paso 4] Procesando recompensa publicitaria (500 puntos)...")
        ad_reward = bank.process_ad_reward(test_user.id, points_amount=500.0)
        print(f"[OK] Puntos otorgados: {ad_reward['points_awarded']:.2f}")
        print(f"    Nuevo saldo: {ad_reward['new_points_balance']:.2f} puntos")
        
        # 5. Mostrar comparación de métodos de retiro (diseño conductual)
        print("\n[Paso 5] Comparando metodos de retiro (300 puntos = $3.00 USD)...")
        print("-" * 60)
        comparison = compare_withdrawal_methods(300.0)
        
        print("\nPayPal:")
        if comparison["paypal"]["available"]:
            print(f"   Comision: {comparison['paypal']['fee_points']:.2f} puntos (10%)")
            print(f"   Neto: {comparison['paypal']['net_points']:.2f} puntos")
            print(f"   Fiat: ${comparison['paypal']['fiat_received']:.2f} USD")
        else:
            print(f"   [ERROR] {comparison['paypal']['error']}")
        
        print("\nKraken Partner:")
        if comparison["kraken"]["available"]:
            print(f"   Comision: {comparison['kraken']['fee_points']:.2f} puntos (0%)")
            print(f"   Bonus: +{comparison['kraken']['bonus_points']:.2f} puntos (5%)")
            print(f"   Total: {comparison['kraken']['net_points']:.2f} puntos")
            print(f"   Fiat: ${comparison['kraken']['fiat_received']:.2f} USD")
        else:
            print(f"   [ERROR] {comparison['kraken']['error']}")
        
        # 6. Procesar retiro a Kraken (incentivo)
        print("\n[Paso 6] Procesando retiro a Kraken Partner (300 puntos = $3.00 USD)...")
        withdrawal = bank.process_withdrawal(
            user_id=test_user.id,
            method=WithdrawalMethod.KRAKEN_PARTNER,
            points_amount=300.0
        )
        print(f"[OK] Retiro procesado exitosamente:")
        print(f"    Puntos retirados: {withdrawal['points_withdrawn']:.2f}")
        print(f"    Bonus recibido: +{withdrawal['bonus_points']:.2f} puntos")
        print(f"    Fiat agregado: ${withdrawal['fiat_added']:.2f} USD")
        print(f"    Puntos restantes: {withdrawal['remaining_points']:.2f}")
        print(f"    Saldo fiat total: ${withdrawal['total_fiat_balance']:.2f} USD")
        
        # 7. Mostrar resumen económico final
        print("\n[Paso 7] Resumen economico del usuario...")
        summary = bank.get_user_economy_summary(test_user.id)
        print(f"    Email: {summary['email']}")
        print(f"    Nexus Points: {summary['nexus_points']:.2f}")
        print(f"    Valor en Fiat (puntos): ${summary['points_fiat_value']:.2f} USD")
        print(f"    Saldo Fiat: ${summary['fiat_balance']:.2f} USD")
        print(f"    Valor total del portfolio: ${summary['total_portfolio_value']:.2f} USD")
        print(f"    Nivel de reputacion: {summary['reputation_level']}")
        
        print("\n" + "=" * 60)
        print("[OK] Demo completada exitosamente!")
        print("\nAnalisis Conductual:")
        print("   - El usuario recibio puntos facilmente (baja friccion)")
        print("   - Al comparar metodos, Kraken es claramente superior")
        print("   - El bonus del 5% crea un 'anclaje' psicologico")
        print("   - El usuario termina con mas puntos de los que retiro (efecto positivo)")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    demo_complete_flow()
