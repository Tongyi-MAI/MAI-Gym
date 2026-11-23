from slime.utils.misc import get_current_node_ip, get_free_port


class RayActor:
    @staticmethod
    def get_runtime_node_id():
        import ray

        return ray.get_runtime_context().get_node_id()

    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
