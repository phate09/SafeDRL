from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
mdp = gateway.entry_point.getMdpSimple()
print(mdp.addState())
print(mdp.addState())
distribution = gateway.entry_point.generateDistribution(0,1)
print(distribution.mean())